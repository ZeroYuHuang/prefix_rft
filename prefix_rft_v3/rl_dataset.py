import copy
import logging
import os
import re
import uuid
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from prefix_rft_v3.scheduler.global_step import BetaSampler, CTRL_MAPPING, CTRL_WRAPPER_MAPPING
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)


def tokenize_and_postprocess_data(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    pad_token_id: int,
    left_pad=True,
    truncation="error",
):
    assert truncation in ["left", "right", "error"]

    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]
    sequence_length = input_ids.shape[-1]

    if sequence_length < max_length:
        input_ids = verl_F.pad_sequence_to_length(
            input_ids,
            max_seq_len=max_length,
            pad_token_id=pad_token_id,
            left_pad=left_pad,
        )
        attention_mask = verl_F.pad_sequence_to_length(
            attention_mask,
            max_seq_len=max_length,
            pad_token_id=0,
            left_pad=left_pad,
        )
    elif sequence_length > max_length:
        if truncation == "left":
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        else:
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")

    return input_ids, attention_mask, sequence_length


def _is_missing_demo_value(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def _extract_demo_text(demo):
    if _is_missing_demo_value(demo):
        return ""
    if isinstance(demo, str):
        return demo
    if isinstance(demo, dict):
        for key in ("raw_response", "demo_raw_response", "response", "completion", "text", "content"):
            value = demo.get(key)
            if isinstance(value, str) and value:
                return value
    return ""


def _extract_demo_correctness(demo, default=True):
    if isinstance(demo, bool):
        return bool(demo)
    if isinstance(demo, dict):
        for key in ("demo_pass", "pass", "passed", "correct", "is_correct", "demo_correct"):
            if key in demo and not _is_missing_demo_value(demo[key]):
                return bool(demo[key])
    return bool(default)


def _ensure_list_of_messages(messages):
    if isinstance(messages, np.ndarray):
        messages = messages.tolist()
    return copy.deepcopy(messages)


class PrefixRFTDataset(Dataset):
    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.max_samples = max_samples

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.max_response_length = config.get("max_response_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.filter_overlong_prompts_length = config.get("filter_overlong_prompts_length", self.max_prompt_length)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.dataset_num_proc = int(config.get("dataset_num_proc", 1))
        if self.dataset_num_proc < 1:
            self.dataset_num_proc = 1
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)

        self.demo_ratio = config.get("demo_ratio", 1.0)
        self.demo_key = config.get("demo_key", "demos")
        self.demo_corr_key = config.get("demo_corr_key", "demos_corr")
        self.total_demo_n = config.get("total_demo_n", 1)
        self.num_prefix = config.get("num_prefix", 1)
        self.num_empty_prefix = config.get("num_empty_prefix", 0)
        self.min_prefix_len = config.get("min_prefix_len", 1)
        self.only_keep_dp_with_demo = config.get("only_keep_dp_with_demo", False)

        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
        self._create_prefix_controller()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        np.random.seed(self.config.get("seed", 1))

        dataframes = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        dataframe = datasets.concatenate_datasets(dataframes)

        total = len(dataframe)
        if self.max_samples > 0 and self.max_samples < total:
            if self.config.get("shuffle", False):
                seed = self.config.get("seed")
                rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            dataframe = dataframe.select(indices.tolist())

        self.dataframe = [dict(example) for example in dataframe]

        print(f"dataset len: {len(self.dataframe)}")
        self._normalize_demo_columns_if_needed()

        if self.filter_overlong_prompts:
            max_prompt_length = self.filter_overlong_prompts_length
            filtered_dataframe = []
            for doc in self.dataframe:
                prompt_text = self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True)
                if len(prompt_text) <= max_prompt_length:
                    filtered_dataframe.append(doc)
            self.dataframe = filtered_dataframe
            print(f"filter dataset len: {len(self.dataframe)}")

        for example in self.dataframe:
            example["avg_rewards"] = []
            example["prefix_ratios"] = []
            if np.random.choice([0, 1], p=[1 - self.demo_ratio, self.demo_ratio]) == 0:
                example[self.demo_key] = []
                if self.demo_corr_key in example:
                    example[self.demo_corr_key] = []

        if self.only_keep_dp_with_demo:
            self.dataframe = [example for example in self.dataframe if len(example[self.demo_key]) > 0]

        for example in self.dataframe:
            example["uid"] = str(uuid.uuid4())

        self.uid2item_index = {}
        for i, example in enumerate(self.dataframe):
            self.uid2item_index[example["uid"]] = i

        self.training_metrics = defaultdict(list)
        self.global_step = 0

    def _normalize_demo_columns_if_needed(self):
        if len(self.dataframe) == 0:
            return

        column_names = set(self.dataframe[0].keys())
        need_normalization = (self.demo_key not in column_names) or (self.demo_corr_key not in column_names)

        if not need_normalization:
            n_probe = min(16, len(self.dataframe))
            for idx in range(n_probe):
                demos = self.dataframe[idx].get(self.demo_key, [])
                if demos:
                    need_normalization = not isinstance(demos[0], str)
                    break

        if not need_normalization:
            return

        print(f"Normalizing demonstration columns into '{self.demo_key}' and '{self.demo_corr_key}'")

        demo_key = self.demo_key
        demo_corr_key = self.demo_corr_key

        for example in self.dataframe:
            demos = []
            demos_corr = []

            raw_demos = example.get(demo_key, [])
            raw_corrs = example.get(demo_corr_key, [])

            if _is_missing_demo_value(raw_demos):
                raw_demos = []
            if _is_missing_demo_value(raw_corrs):
                raw_corrs = []

            if raw_demos:
                for idx, demo in enumerate(raw_demos):
                    demo_text = _extract_demo_text(demo)
                    if not demo_text:
                        continue
                    demos.append(demo_text)
                    if idx < len(raw_corrs) and not _is_missing_demo_value(raw_corrs[idx]):
                        demos_corr.append(bool(raw_corrs[idx]))
                    else:
                        demos_corr.append(_extract_demo_correctness(demo, default=True))

            if not demos:
                for source_key, default_corr in (("demonstrations", True), ("unpassed_demonstrations", False)):
                    source_demos = example.get(source_key, [])
                    if _is_missing_demo_value(source_demos):
                        source_demos = []
                    for demo in source_demos:
                        demo_text = _extract_demo_text(demo)
                        if not demo_text:
                            continue
                        demos.append(demo_text)
                        demos_corr.append(_extract_demo_correctness(demo, default=default_corr))

            if not demos:
                demo_text = _extract_demo_text(example.get("demo_raw_response", ""))
                if demo_text:
                    demos.append(demo_text)
                    demos_corr.append(bool(example.get("demo_pass", True)))

            example[demo_key] = demos
            example[demo_corr_key] = demos_corr

    def _get_column_values(self, key):
        return [example.get(key) for example in self.dataframe]

    def _create_prefix_controller(self):
        prefix_low_ctrl_cls = CTRL_MAPPING[self.config.prefix_low_ctrl_type]
        self.prefix_low_ctrl = prefix_low_ctrl_cls(**self.config.prefix_low_ctrl.kwargs)

        prefix_low_ctrl_wrapper_type = self.config.get("prefix_low_ctrl_wrapper_type", None)
        if prefix_low_ctrl_wrapper_type:
            prefix_low_ctrl_wrapper_cls = CTRL_WRAPPER_MAPPING[prefix_low_ctrl_wrapper_type]
            self.prefix_low_ctrl = prefix_low_ctrl_wrapper_cls(
                self.prefix_low_ctrl, **self.config.prefix_low_ctrl_wrapper.kwargs
            )

        prefix_high_ctrl_cls = CTRL_MAPPING[self.config.prefix_high_ctrl_type]
        self.prefix_high_ctrl = prefix_high_ctrl_cls(**self.config.prefix_high_ctrl.kwargs)

        prefix_high_ctrl_wrapper_type = self.config.get("prefix_high_ctrl_wrapper_type", None)
        if prefix_high_ctrl_wrapper_type:
            prefix_high_ctrl_wrapper_cls = CTRL_WRAPPER_MAPPING[prefix_high_ctrl_wrapper_type]
            self.prefix_high_ctrl = prefix_high_ctrl_wrapper_cls(
                self.prefix_high_ctrl, **self.config.prefix_high_ctrl_wrapper.kwargs
            )

        self.prefix_controller = BetaSampler(
            low_ctrl=self.prefix_low_ctrl,
            high_ctrl=self.prefix_high_ctrl,
            **self.config.prefix_sampler.kwargs,
        )

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages = _ensure_list_of_messages(example[self.prompt_key])

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                if not isinstance(content, str):
                    continue
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    elif segment:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list

        return messages

    def _get_prefix_ratio(self, idx):
        return self.prefix_controller.value(
            avg_rewards=self._get_column_values("avg_rewards"),
            prefix_ratios=self._get_column_values("prefix_ratios"),
            data_idx=idx,
            global_step=self.global_step,
            train_reward=self.training_metrics["train_rewards"],
        )

    def _get_prefix(self, text, prefix_ratio):
        tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        demo_len = len(tokens)
        if prefix_ratio == 0 or demo_len == 0:
            return "", 0, demo_len
        prefix_len = int(prefix_ratio * demo_len)
        if prefix_len < self.min_prefix_len:
            prefix_len = self.min_prefix_len
        if prefix_len >= demo_len:
            prefix_len = demo_len - 1
        prefix = self.tokenizer.decode(tokens[:prefix_len])
        return prefix, prefix_len, demo_len

    def _render_prompt(self, messages, prefix=""):
        if not prefix:
            return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        chat_with_prefix = copy.deepcopy(messages)
        chat_with_prefix.append({"role": "assistant", "content": prefix})
        try:
            return self.tokenizer.apply_chat_template(chat_with_prefix, continue_final_message=True, tokenize=False)
        except TypeError:
            logger.warning("Tokenizer does not support continue_final_message; falling back to default rendering.")
            return self.tokenizer.apply_chat_template(chat_with_prefix, tokenize=False)

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        row_dict["raw_prompt_lengths"] = int(
            self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[-1]
        )

        demos = row_dict.get(self.demo_key, [])[: self.total_demo_n]
        if len(demos) == 0:
            demos = ["" for _ in range(self.total_demo_n)]

        demos_corr = row_dict.get(self.demo_corr_key, [])[: self.total_demo_n]
        if len(demos_corr) == 0:
            demos_corr = [True for _ in range(len(demos))]

        prompt_variants = []
        num_empty_prefix = 0

        for i in range(self.num_prefix):
            if self.num_empty_prefix > num_empty_prefix:
                prefix_ratio, prefix_ctrl_low, prefix_ctrl_high = 0.0, -1, -1
                num_empty_prefix += 1
            else:
                prefix_ratio, prefix_ctrl_low, prefix_ctrl_high = self._get_prefix_ratio(item)

            demo_selected_idx = int(np.random.choice([j for j in range(len(demos))]))
            demo = demos[demo_selected_idx]
            demo_corr = demos_corr[demo_selected_idx]

            prefix, prefix_len, demo_len = self._get_prefix(demo, prefix_ratio)
            prompt_with_chat_template = self._render_prompt(messages, prefix=prefix)
            prompt_variants.append(prompt_with_chat_template)

            input_ids, attention_mask, sequence_length = tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )

            if sequence_length > self.max_prompt_length and 0 < prefix_ratio < 1:
                prefix_len -= sequence_length - self.max_prompt_length
                prefix_len = max(prefix_len, self.min_prefix_len)

            position_ids = compute_position_id_with_mask(attention_mask)
            normalized_prefix_ratio = prefix_len / demo_len if demo_len > 0 else 0.0

            row_dict[f"input_ids_{i}"] = input_ids[0]
            row_dict[f"attention_mask_{i}"] = attention_mask[0]
            row_dict[f"position_ids_{i}"] = position_ids[0]
            row_dict[f"prefix_ratio_{i}"] = normalized_prefix_ratio
            row_dict[f"prefix_uid_{i}"] = str(uuid.uuid4()) if normalized_prefix_ratio > 0 else "0000-0000-0000-0000"
            row_dict[f"prefix_correct_{i}"] = bool(demo_corr)
            row_dict[f"demo_len_{i}"] = int(demo_len)
            row_dict[f"prefix_ctrl_low_{i}"] = prefix_ctrl_low
            row_dict[f"prefix_ctrl_high_{i}"] = prefix_ctrl_high

        row_dict["input_ids"] = row_dict["input_ids_0"]
        row_dict["attention_mask"] = row_dict["attention_mask_0"]
        row_dict["position_ids"] = row_dict["position_ids_0"]
        row_dict["prefix_ratio"] = row_dict["prefix_ratio_0"]
        row_dict["prefix_uid"] = row_dict["prefix_uid_0"]
        row_dict["prefix_correct"] = row_dict["prefix_correct_0"]

        raw_prompt_ids = self.tokenizer.encode(prompt_variants[0], add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = prompt_variants

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()
