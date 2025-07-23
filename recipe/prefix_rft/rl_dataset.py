# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

from verl.utils.model import compute_position_id_with_mask
# from recipe.uni_rft.scheduler.global_step import *
from recipe.prefix_rft_v2.scheduler.global_step import *
import verl.utils.torch_functional as verl_F
import uuid


def tokenize_and_postprocess_data(prompt: str,
                                  tokenizer: PreTrainedTokenizer,
                                  max_length: int,
                                  pad_token_id: int,
                                  left_pad=True,
                                  truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = verl_F.pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = verl_F.pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            print(f"Right truncated! Because the sequence length {sequence_length} is longer than {max_length} May truncate the final answer")
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask, sequence_length


def collate_fn(data_list: list[dict]) -> dict:
    """Collate a batch of data."""
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def is_continue_final_message(messages: List[dict]):
    return messages[-1]["role"] == "assistant"
    
class PrefixRFTDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_overlong_prompts_length=config.get("filter_overlong_prompts_length", 1024)
        self.demo_ratio = config.get("demo_ratio", 1.0)
        self.demo_key = config.get('demo_key', 'demos')
        self.demo_corr_key = config.get('demo_corr_key', 'demos_corr')
        self.max_response_length = config.max_response_length
        self.total_demo_n = config.total_demo_n

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

        np.random.seed(self.config.seed)

        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.filter_overlong_prompts_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.filter_overlong_prompts_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")
        # initialize historical avg_reward with empty list
        # 使用 map 方法为每行添加空列表
        def initialize_columns(example):
            example["avg_rewards"] = []
            example["prefix_ratios"] = []
            return example

        def delete_demos(example):
            if np.random.choice([0, 1], p=[1-self.demo_ratio, self.demo_ratio]) == 0:
                example[self.demo_key] = []
                if self.demo_corr_key in example:
                    example[self.demo_corr_key] = []
            return example

        self.dataframe = self.dataframe.map(initialize_columns, num_proc=16)
        print(f"Only {self.demo_ratio} of the entire dataset has the demos")
        self.dataframe = self.dataframe.map(delete_demos, num_proc=16)

        if self.config.only_keep_dp_with_demo:
            print("Only keep the data points with demonstrateion answers")
            self.dataframe = self.dataframe.filter(lambda x: len(x[self.demo_key]) > 0)

        # add unique ids for each datapoint
        # 使用 map 方法为每行生成唯一 ID
        def add_unique_id(example):
            example["uid"] = str(uuid.uuid4())
            return example

        self.dataframe = self.dataframe.map(add_unique_id, num_proc=16)

        # construct the mapping for uid2item_index
        # 遍历 Dataset 构建 uid 到索引的映射
        self.uid2item_index = {}
        for i, example in enumerate(self.dataframe):
            uid = example['uid']
            self.uid2item_index[uid] = i

        # dict to save training metrics
        self.training_metrics = defaultdict(list)
        self.global_step = 0
        

    def _create_prefix_controller(self):

        print("Creating prefix ctrls")

        prefix_low_ctrl_cls = CTRL_MAPPING[self.config.prefix_low_ctrl_type]
        self.prefix_low_ctrl = prefix_low_ctrl_cls(**self.config.prefix_low_ctrl.kwargs)

        prefix_low_ctrl_wrapper_type = self.config.get("prefix_low_ctrl_wrapper_type", None)
        if prefix_low_ctrl_wrapper_type:
            prefix_low_ctrl_wrapper_cls = CTRL_WRAPPER_MAPPING[prefix_low_ctrl_wrapper_type]
            self.prefix_low_ctrl = prefix_low_ctrl_wrapper_cls(self.prefix_low_ctrl, **self.config.prefix_low_ctrl_wrapper.kwargs)

        print(f"Prefix low ctrl used: {self.prefix_low_ctrl}")

        prefix_high_ctrl_cls = CTRL_MAPPING[self.config.prefix_high_ctrl_type]
        self.prefix_high_ctrl = prefix_high_ctrl_cls(**self.config.prefix_high_ctrl.kwargs)

        prefix_high_ctrl_wrapper_type = self.config.get("prefix_high_ctrl_wrapper_type", None)
        if prefix_high_ctrl_wrapper_type:
            prefix_high_ctrl_wrapper_cls = CTRL_WRAPPER_MAPPING[prefix_high_ctrl_wrapper_type]
            self.prefix_high_ctrl = prefix_high_ctrl_wrapper_cls(self.prefix_high_ctrl, **self.config.prefix_high_ctrl_wrapper.kwargs)
        
        print(f"Prefix high ctrl used: {self.prefix_high_ctrl}")
        
        self.prefix_controller = BetaSampler(
            low_ctrl=self.prefix_low_ctrl, 
            high_ctrl=self.prefix_high_ctrl,
            **self.config.prefix_sampler.kwargs 
        )

        print(f"Final Prefix controller (sampler): {self.prefix_controller}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        return messages


    def _get_prefix_ratio(self, idx):
        return self.prefix_controller.value(
            avg_rewards=self.dataframe["avg_rewards"],
            prefix_ratios=self.dataframe["prefix_ratios"],
            data_idx=idx,
            global_step=self.global_step,
            train_reward=self.training_metrics["train_rewards"]
        )

    def _get_prefix(self, text, prefix_ratio):
        tokens = self.tokenizer(text)["input_ids"]
        demo_len = len(tokens)
        if prefix_ratio == 0 or demo_len == 0:
            return "", 0, demo_len
        prefix_len = int(prefix_ratio * len(tokens))
        # for training stability
        if prefix_len < self.config.min_prefix_len:
            prefix_len = self.config.min_prefix_len 
        
        # if prefix_len > self.config.max_prefix_len:
        #     prefix_len = self.config.max_prefix_len
        
        if prefix_len >= demo_len:
            prefix_len = demo_len - 1
        
        return self.tokenizer.decode(tokens[:prefix_len]), prefix_len, demo_len

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # length of paw prompt
        row_dict["raw_prompt_lengths"] = int(self.tokenizer(raw_prompt, return_tensors='pt', add_special_tokens=False)['input_ids'].shape[-1])
        # -------------------------------
        # Step 0: suppose we have multiple demonstration data for each data point, we random sample one
        # -------------------------------
        demos = []
        if self.demo_key in row_dict:
            demos = row_dict[self.demo_key][: self.total_demo_n]
        if len(demos) == 0:
            print("The datapoint does not have demos. Use empty string as place holders")
            demos = ["" for _ in range(self.total_demo_n)] # placeholders

        demos_corr = []
        if self.demo_corr_key in row_dict:
            demos_corr = row_dict[self.demo_corr_key][: self.total_demo_n]
        if len(demos_corr) == 0:
            demos_corr = [True for _ in range(len(demos))]
        
        # -------------------------------
        # Step1.1: prepare prompts + prefix (the prefix could  be empty) for RL generation
        # -------------------------------
        # there are self.total_demon_n demostrations in total, and we randomly sample 1
        num_empty_prefix, prefix_ratios = 0, []
        for i in range(self.config.num_prefix):
            if self.config.num_empty_prefix > num_empty_prefix:
                prefix_ratio, prefix_ctrl_low, prefix_ctrl_high = 0, -1, -1
                num_empty_prefix += 1
            else:
                # get prefix ratio
                prefix_ratio, prefix_ctrl_low, prefix_ctrl_high = self._get_prefix_ratio(item)
            # Radonly sample one demonstration, so that each prefix can from different deomnstration data
            demo_selected_idx = np.random.choice([_ for _ in range(len(demos))])
            demo = demos[demo_selected_idx]
            demo_corr = demos_corr[demo_selected_idx]
            prefix, prefix_len, demo_len = self._get_prefix(demo, prefix_ratio)  # this prefix will be later used fot sft loss training
            if prefix_ratio == 0 or prefix_len == 0:
                prompt_with_chat_template = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                assert prefix_len != 0
                chat_with_prefix = list(messages)
                chat_with_prefix.append({"role": "assistant", "content": prefix})
                prompt_with_chat_template = self.tokenizer.apply_chat_template(chat_with_prefix, continue_final_message=True, tokenize=False)
            
            raw_prompt = prompt_with_chat_template  # with chat template

            input_ids, attention_mask, sequence_length = tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right')
            
            # if the prefix_len + prompt_length is bigger than the max_prompt_length, the prefix should be truncated
            if sequence_length > self.max_prompt_length and 0 < prefix_ratio < 1:
                prefix_len -= (sequence_length - self.max_prompt_length)
                assert prefix_len > self.config.min_prefix_len
            position_ids = compute_position_id_with_mask(attention_mask)

            row_dict[f'input_ids_{i}'] = input_ids[0]
            row_dict[f'attention_mask_{i}'] = attention_mask[0]
            row_dict[f'position_ids_{i}'] = position_ids[0]
            prefix_ratio = prefix_len / demo_len if demo_len > 0 else 0
            prefix_ratios.append(prefix_ratio)
            row_dict[f'prefix_ratio_{i}'] = prefix_ratio
            row_dict[f'prefix_uid_{i}'] = str(uuid.uuid4()) if prefix_ratio > 0 else '0000-0000-0000-0000'
            row_dict[f'prefix_correct_{i}'] = demo_corr
            row_dict[f'demo_len_{i}'] = demo_len 
            row_dict[f'prefix_ctrl_low_{i}'] = prefix_ctrl_low
            row_dict[f'prefix_ctrl_high_{i}'] = prefix_ctrl_high
        # row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        # row_dict["prefix_ratio"] = np.mean(prefix_ratios)
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
