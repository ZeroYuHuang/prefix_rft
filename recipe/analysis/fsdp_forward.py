import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import json
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
# from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import CPUOffloadPolicy, MixedPrecisionPolicy, apply_fsdp2, fsdp2_clip_grad_norm_, fsdp2_load_full_state_dict, get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

import os
import torch.distributed
from verl.utils.device import get_nccl_backend, get_torch_device
from typing import List, Union


import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
import torch.distributed as dist

def initialize_global_process_group(timeout_second=36000):
    from datetime import timedelta

    torch.distributed.init_process_group(get_nccl_backend(), timeout=timedelta(seconds=timeout_second))
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        get_torch_device().set_device(local_rank)
    return local_rank, rank, world_size

def destroy_global_process_group():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# --- Helper Functions ---
def compute_position_id_with_mask(attention_mask: torch.Tensor):
    """
    Helper function to compute position_ids from an attention mask.
    This is a placeholder implementation based on standard causal LM assumptions.
    """
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


# --- Dataset Class (Modified from user's version) ---
class ForwardDataset(Dataset):
    """
    This is an in-memory SFTDataset, modified to support metrics calculation.
    
    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        # ... (Configuration from user's script)
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "error")
        self.use_shm = config.get("use_shm", False)
        
        # Ensure parquet_files is a list
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files] if parquet_files else []
        self.parquet_files = parquet_files

        # Load and process data
        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        # ... (Pandas processing logic from user's script)
        # dataframes = [pd.read_parquet(pf) for pf in self.parquet_files]
        # self.dataframe = pd.concat(dataframes).reset_index(drop=True)
        from datasets import load_dataset
        self.dataset = load_dataset("parquet", data_files=self.parquet_files)['train']
        # Ensure 'uid' column exists
        # if 'uid' not in self.dataframe.columns:
        #     logger.warning("'uid' column not found in data. Generating sequential UIDs.")
        #     raise NotImplementedError

        # flat the dataset
        self.prompts, self.responses, self.uids, self.response_ids = [], [], [], []
        num_data = len(self.dataset)
        for i in range(num_data):
            dp = self.dataset[i]
            all_demos = dp["old_demos"]
            for j in range(len(all_demos)):
                self.prompts.append(dp["prompt"])
                self.responses.append(all_demos[j])
                self.uids.append(dp["uid"])
                self.response_ids.append(str(j))
        
        # self.prompts = self.dataframe["prompt"].tolist()
        # self.responses = self.dataframe["response"].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        # --- ADDED: Get the UID ---
        uid = self.uids[item]
        prompt = self.prompts[item]
        response = self.responses[item]
        response_id = self.response_ids[item]
        
        # Here we assume that it's already the message list for chat_temnlate
        prompt_chat = prompt
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        # --- THIS IS THE KEY VARIABLE WE NEED ---
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")
        
        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0
        return {
            "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "uid": uid, "answer_start_index": prompt_length, "loss_mask": loss_mask,
            "response_id": response_id
        }


# --- Main Application Class ---
class FSDPForwardCalculator:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.device_name = get_device_name()

        self._build_dataloader(dataset)
        self._build_model()

        if self.device_mesh.get_rank() == 0:
            logger.info("Configuration:\n%s", self.config)

    def _build_dataloader(self, dataset: Dataset):
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if rank == 0:
            logger.info(f"Using FSDP rank {rank} and size {world_size} for data distribution.")
        print(f"The length of the dataset {len(dataset)}")
        sampler = DistributedSampler(dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        trust_remote_code = self.config.model.get("trust_remote_code", False)
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        # Use meta device to avoid allocating large model weights in CPU RAM unnecessarily
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)
        
        with init_context():
            self.model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
        
        # FSDP setup
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch_dtype)
        auto_wrap_policy = get_fsdp_wrap_policy(self.model, config=self.config.model.fsdp_config.wrap_policy)
        cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
        fsdp_kwargs = {
            "mesh": self.device_mesh,
            "mp_policy": mp_policy,
            "offload_policy": None,
            "reshard_after_forward": True,
        }
        full_state = self.model.state_dict()
        apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
        fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
        self.fsdp_model = self.model

    def _calculate_for_one_batch(self, batch):    
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        labels = input_ids[:, 1:].contiguous()
        outputs = self.fsdp_model(
            input_ids=input_ids, attention_mask=attention_mask, 
            position_ids=position_ids, use_cache=False
        )
        logits = outputs.logits.float()
        batch_results = []
        for i in range(len(batch["uid"])):
            uid = batch["uid"][i]
            answer_start = batch["answer_start_index"][i]
            sequence_end = attention_mask[i].sum().item() - 1

            answer_logits = logits[i, answer_start - 1 : sequence_end, :]
            answer_token_ids = input_ids[i, answer_start : sequence_end + 1]
            # token log prob
            log_probs_dist = F.log_softmax(answer_logits, dim=-1)
            token_log_probs = torch.gather(log_probs_dist, -1, answer_token_ids.unsqueeze(-1)).squeeze(-1)
            # token entropy
            entropy_dist = torch.distributions.Categorical(logits=answer_logits)
            token_entropies = entropy_dist.entropy()
            # ppl
            avg_neg_log_likelihood = -token_log_probs.mean()
            perplexity = torch.exp(avg_neg_log_likelihood)
            # token level loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels.contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss * loss_mask.to(loss.device)
            final_loss = loss.sum() / loss_mask.sum()
            batch_results.append({
                "uid": uid,
                "token_log_p": token_log_probs.cpu().tolist(),
                "token_entropy": token_entropies.cpu().tolist(),
                "perplexity": perplexity.cpu().item(),
                "loss": loss.cpu().tolist(),
                "loss_mask": loss_mask.cpu().tolist(),
                "final_loss": final_loss.cpu().item()
            })
        return batch_results
    
    def run(self):
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        
        self.fsdp_model.eval()
        local_results = []

        for data in tqdm(self.dataloader, desc="Calculating Metrics"):
            data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
            with torch.no_grad():
                batch_results = self._calculate_for_one_batch(data)
                local_results.extend(batch_results)
        
        torch.distributed.barrier()
        if world_size > 1:
            if rank == 0: logger.info(f"Gathering results from all {world_size} ranks...")
            all_process_results = [None] * world_size
            torch.distributed.gather_object(local_results, all_process_results if rank == 0 else None, dst=0)
        else:
            all_process_results = [local_results]

        if rank == 0:
            final_results = [item for sublist in all_process_results for item in sublist]
            output_file = self.config.output_file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                for res in final_results:
                    f.write(json.dumps(res) + '\n')
            logger.info(f"✅ Metrics calculation complete. {len(final_results)} results saved to {output_file}")
        torch.distributed.barrier()


def forward(config):
    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.partial_pretrain,
        trust_remote_code=config.model.get("trust_remote_code", False)
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer pad_token set to eos_token.")

    dataset = ForwardDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        config=config.data
    )
    print(dataset)

    calculator = FSDPForwardCalculator(config=config, device_mesh=device_mesh, tokenizer=tokenizer, dataset=dataset)
    calculator.run()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="forward", version_base=None)
def main(config):
    forward(config)


if __name__ == "__main__":

    main()