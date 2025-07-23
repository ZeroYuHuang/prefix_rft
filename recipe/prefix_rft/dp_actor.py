# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import itertools
import logging
import os
from typing import Tuple

import torch
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from recipe.prefix_rft_v2.core_algos import agg_loss, compute_policy_loss, kl_penalty
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor

from recipe.prefix_rft_v2.scheduler.global_step import *

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        print(f"Actor use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

        if self.use_fused_kernels:
            from verl.utils.experimental.torch_functional import FusedLinearForPPO

            self.fused_linear_for_ppo = FusedLinearForPPO()

            # FusedLinearForPPO has an error when compiled, disable for now
            # if self.config.get("use_torch_compile", True):
            #     self.fused_linear_for_ppo.compile(dynamic=True)
        
        # prepare for reshaping 
        reshapers = self.config.get("off_adv_reshaper", None)
        self.reshapers = reshapers.split(",") if "," in reshapers else [reshapers]
        print(f"The reshapers are as follow: {self.reshapers}")
        if "entropy" in self.reshapers or \
            "entropy_low" in self.reshapers or \
                "random" in self.reshapers:
            print("Creating entropy masking reshaper")
            ent_ctrl_cls = CTRL_MAPPING[self.config.off_ent_mask_ratio_ctrl_type]
            self.ent_mask_ratio_ctrl = ent_ctrl_cls(**self.config.off_ent_mask_ratio_ctrl.kwargs)
            print(f"The ent mask reshaper used is: {self.ent_mask_ratio_ctrl}")

    def reshape_func(
            self,
            adv,
            log_prob,
            prefix_mask,
            **kwargs
        ):
        global_step=kwargs["global_step"]
        prefix_mask = prefix_mask.bool()
        prefix_adv = adv[prefix_mask]
        if prefix_adv.numel() == 0:
            # no prefix, return the original prefix
            return adv, log_prob

        reshapers = self.reshapers

        def identity(adv, log_prob):
            return adv, log_prob

        def importance_sampling(adv, log_prob):
            print("Doing importance sampling by adv reshaping")
            old_log_prob = kwargs.get("old_log_prob")
            log_prob[prefix_mask] = log_prob[prefix_mask] + old_log_prob[prefix_mask]
            return adv, log_prob

        def luffy(adv, log_prob):
            print("Doing luffy reshaping")
            old_log_prob = kwargs.get("old_log_prob")
            log_prob[prefix_mask] = log_prob[prefix_mask] + old_log_prob[prefix_mask] - torch.log(torch.exp(log_prob[prefix_mask]) + 0.1)
            return adv, log_prob

        def silu(adv, log_prob):
            print("Doing silu reshaping")
            adv[prefix_mask] = torch.nn.functional.silu(adv[prefix_mask])
            return adv, log_prob

        def relu(adv, log_prob):
            print("Doing relu reshaping")
            adv[prefix_mask] = torch.nn.functional.relu(adv[prefix_mask])
            return adv, log_prob

        def entropy(adv, log_prob):            
            device = adv.device
            print("Doing entropy-based reshaping")
            entropy = kwargs.get("entropy", None)
            assert entropy is not None

            entropy = entropy[prefix_mask]
            _, indices = torch.sort(entropy)
            # entropy_mask_ratio = self.config.reshape_kwargs.get.entropy_mask_ratio
            entropy_mask_ratio = self.ent_mask_ratio_ctrl.value(global_step=global_step)
            split_point = int(len(indices) * entropy_mask_ratio)
            # 获取较小entropy的索引（即排序后靠前的一半）
            bottom_n_percent_indices = indices[:split_point]  # 这些是 entropy 较小的样本索引

            adv_mask = torch.ones_like(adv, dtype=torch.bool)
            arange_tensor = torch.arange(len(adv[prefix_mask]))
            isin_mask = torch.isin(arange_tensor, bottom_n_percent_indices).to(device)

            # 更新 adv_mask[prefix_mask]
            adv_mask[prefix_mask] &= ~isin_mask

            # 将不符合条件的 adv 设为 0
            adv[~adv_mask] = 0
            print(f"Doing entropy-based reshaping with entropy split equals to {entropy_mask_ratio}, with {torch.sum(~adv_mask)} tokens from {torch.sum(prefix_mask)}adv are set to 0.")

            return adv, log_prob  # 假设 data 中有 log_prob


        def entropy_low(adv, log_prob):
            device = adv.device
            print("Doing entropy-based reshaping")
            entropy = kwargs.get("entropy", None)
            assert entropy is not None

            entropy = entropy[prefix_mask]
            _, indices = torch.sort(entropy)
            # entropy_mask_ratio = self.config.reshape_kwargs.get.entropy_mask_ratio
            entropy_mask_ratio = self.ent_mask_ratio_ctrl.value(global_step=global_step)
            split_point = int(len(indices) * entropy_mask_ratio)
            # 获取后50%的索引（即排序后靠前的一半）
            bottom_n_percent_indices = indices[:split_point]  # 这些是 entropy 较小的样本索引

            adv_mask = torch.ones_like(adv, dtype=torch.bool)
            arange_tensor = torch.arange(len(adv[prefix_mask]))
            isin_mask = torch.isin(arange_tensor, bottom_n_percent_indices).to(device)

            # 更新 adv_mask[prefix_mask]
            adv_mask[prefix_mask] &= ~isin_mask

            # 将不符合条件的 adv 设为 0
            adv[adv_mask] = 0
            print(f"Doing entropy-based reshaping with entropy split equals to {entropy_mask_ratio}, with {torch.sum(~adv_mask)} tokens from {torch.sum(prefix_mask)}adv are set to 0.")

            return adv, log_prob  # 假设 data 中有 log_prob

        def position(adv, log_prob, eps=1e-3):
            prefix_relpos = kwargs.get("prefix_relpos", None)
            a = kwargs.get("prefix_ctrl_low", None)
            b = kwargs.get("prefix_ctrl_high", None)
            # print(prefix_relpos, a, b)
            assert prefix_relpos is not None and a is not None and b is not None
            denom = b ** 2 - a ** 2
            f = torch.where(
                prefix_relpos <= a,
                2.0 / (a + b),
                torch.where(
                    prefix_relpos <= b,
                    2.0 * (b - prefix_relpos) / denom,
                    0.0
                )
            )
            # doing reverse
            w = 1.0 / (f.clamp_min(eps))
            # print(w[prefix_mask], torch.max(w[prefix_mask]), torch.min(w[prefix_mask]))
            mean_w = w[prefix_mask].sum() / prefix_mask.sum()
            w_hat = w / mean_w
            # print(w_hat[prefix_mask], torch.max(w_hat[prefix_mask]), torch.min(w_hat[prefix_mask]))
            adv[prefix_mask] = w_hat[prefix_mask] * adv[prefix_mask]
            return adv, log_prob


        def random_mask(adv, log_prob):
            """
            在 prefix_mask 区域内，随机选择 n% 的样本，
            将这些样本对应的 adv 设置为 0。

            Args:
                adv (Tensor): 奖励值张量 [batch_size, seq_len]
                log_prob (Tensor): 日志概率张量（未使用，保留接口）
                n (int): 要 mask 掉的随机 n% 的样本比例

            Returns:
                adv, log_prob: 修改后的 adv 和原始 log_prob
            """
            n = self.ent_mask_ratio_ctrl.value(global_step=global_step)
            print(f"Doing random masking: masking bottom {n} of prefix tokens")

            # 获取 prefix_mask 区域的设备信息
            device = adv.device

            # 获取 prefix_mask 中 True 的位置
            prefix_indices = torch.where(prefix_mask)

            # 总样本数
            total_samples = len(prefix_indices[0])
            num_to_mask = int(total_samples * n / 100)

            # 随机打乱索引并选取前 num_to_mask 个
            rand_indices = torch.randperm(total_samples)[:num_to_mask]

            # 获取要 mask 的样本在原始 tensor 中的位置
            mask_positions = tuple(idx[rand_indices] for idx in prefix_indices)

            # 将这些位置的 adv 设为 0
            adv[mask_positions] = 0

            return adv, log_prob

        def length_on(adv, log_prob):
            """
            This function adjusts the advantages for sequences that reach the maximum sequence length.
            If a response sequence fills the entire available length (i.e., hits max_seq_length),
            the advantage for the on-policy part of that sequence (the part after the prefix)
            is set to zero. This is a technique to mitigate potential issues from truncated sequences
            where the model is forced to stop generating, which is not a natural completion.

            Args:
                adv (torch.Tensor): The advantage tensor.
                log_prob (torch.Tensor): The log probability tensor.
                **kwargs: A dictionary of keyword arguments. It must contain 'response_mask' and 'prefix_mask'.

            Returns:
                torch.Tensor: The modified advantage tensor.
            """
            response_mask = kwargs.get("response_mask", None)

            assert response_mask is not None, "response_mask must be provided in kwargs"

            device = adv.device
            max_seq_length = response_mask.shape[-1]
            # Calculate the length of each response by summing the response_mask
            response_length = response_mask.sum(dim=-1)

            # Identify sequences where the response length reaches the maximum possible length.
            # These are the sequences that were likely truncated.
            end_hit_mask = (response_length == max_seq_length).float()

            # We want to zero out the advantage for the generated part (on-policy) of these truncated sequences.
            # The on-policy part is the response, which is indicated by the response_mask.
            # However, we must exclude the prompt/prefix part, so we use `(1 - prefix_mask)`.
            on_policy_mask = (~prefix_mask.to(device)) * response_mask

            # Create a mask for the advantages that need to be zeroed out.
            # It applies to sequences that hit the max length (`end_hit_mask.unsqueeze(-1)`)
            # and only for the on-policy tokens (`on_policy_mask`).
            zero_adv_mask = end_hit_mask.unsqueeze(-1) * on_policy_mask

            # Set the advantage to zero for the identified tokens.
            adv = adv * (1 - zero_adv_mask)

            print(f"Doing length_on reshaping: if the response hits the max_seq_length, its on policy part (not in the prefix)'s advantage will be set as zero, mask_ratio={torch.mean(end_hit_mask)}")
            print(f"mask_ratio={torch.mean(end_hit_mask)}, max_seq_length={max_seq_length}, end_hit_mask.size()={end_hit_mask.size()}, resopnse_mask_size={response_mask.size()}")

            return adv, log_prob

        reshape_fn_mapping = {
            'identity': identity, 
            'importance_sampling': importance_sampling,
            'luffy': luffy, 'silu': silu, 'relu': relu,
            "entropy": entropy, 
            "entropy_low": entropy_low,
            "random": random_mask,
            "position": position,
            "length_on": length_on 
            # "old_entropy": old_entropy,
            # 'entropy': entropy, 'equal': equal, 
        }

        for r in reshapers:
            if r in reshape_fn_mapping:
                print(f"Applying {r}rehsper func")
                reshape_fn = reshape_fn_mapping[r]
                adv, log_prob = reshape_fn(adv, log_prob)
            else:
                print(f"{r} is not in the rehsaping_fn_mapping")
        
        return adv, log_prob

    def _forward_micro_batch(self, micro_batch, temperature, calculate_entropy=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad=position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.actor_module.lm_head.weight

                    log_probs, entropy_rmpad = self.fused_linear_for_ppo(
                        hidden_states=hidden_states.squeeze(0),
                        vocab_weights=vocab_weights,
                        input_ids=input_ids_rmpad_rolled,
                        temperature=temperature,
                    )

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                    # logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    hidden_states = output.last_hidden_state
                    vocab_weights = self.actor_module.lm_head.weight

                    log_probs, entropy = self.fused_linear_for_ppo(
                        hidden_states=hidden_states[:, -response_length - 1 : -1, :],
                        vocab_weights=vocab_weights,
                        input_ids=micro_batch["responses"],
                        temperature=temperature,
                    )

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature, calculate_entropy=calculate_entropy)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        multi_turn = data.meta_info.get("multi_turn", False)
        global_step = data.meta_info["global_step"]
        prefix_ctrl_low = data.meta_info["prefix_ctrl_low"]
        prefix_ctrl_high = data.meta_info["prefix_ctrl_high"]

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids", 
            "old_log_probs", # "old_entropy", 
            "advantages", "prefix_mask", "prefix_relpos"
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                    print(f"max token len is: {self.config.ppo_max_token_len_per_gpu}, after split we get: {len(micro_batches)} micro_batches, the total_seq_len is: {mini_batch['attention_mask'].sum().item()}")
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for mini_batch_idx, data in enumerate(micro_batches):  # 这里的micto_batch经过了rearrange
                    print(f"MICROBATCH STEP: {mini_batch_idx}")
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    if multi_turn:
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]

                    old_log_prob = data["old_log_probs"]
                    # old_entropy = data["old_entropy"]
                    advantages = data["advantages"]
                    prefix_mask = data["prefix_mask"]
                    prefix_relpos = data["prefix_relpos"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_low_off = self.config.get("clip_ratio_low_off", None)
                    clip_ratio_high_off = self.config.get("clip_ratio_high_off", None)
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    enable_clip = self.config.enable_clip

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0 or "entropy" in self.reshapers:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature, calculate_entropy=calculate_entropy)
                    
                    # 这里的reshape的是一个mini batch离的
                    advantages, log_prob = self.reshape_func(
                        adv=advantages, 
                        log_prob=log_prob, 
                        prefix_mask=prefix_mask.cpu(), 
                        old_log_prob=old_log_prob, 
                        entropy=entropy.cpu().detach(),
                        # old_entropy=old_entropy.cpu().detach(),
                        global_step=global_step,
                        prefix_relpos=prefix_relpos,
                        prefix_ctrl_low=prefix_ctrl_low,
                        prefix_ctrl_high=prefix_ctrl_high,
                        response_mask=response_mask,
                    )

                    if self.config.get("replace_old_log_prob", False):
                        print("Replacing old log prob with newly calculated prob")
                        old_log_prob[prefix_mask] = log_prob[prefix_mask].detach()

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_prefix, pg_clipfrac_lower_prefix = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        prefix_mask=prefix_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        cliprange_low_off=clip_ratio_low_off,
                        cliprange_high_off=clip_ratio_high_off,
                        clip_ratio_c=clip_ratio_c,
                        enable_clip=enable_clip,
                        loss_agg_mode=loss_agg_mode,
                        max_tokens=self.config.loss_agg_max_tokens)
                    # compute entropy loss from entropy
                    if calculate_entropy:
                        if self.config.entropy_mode == "response_only":
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask & (~prefix_mask), loss_agg_mode="token-mean")
                        else:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode="token-mean")

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        entropy_loss = torch.zeros_like(pg_loss)
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        print(f"original with dynamic bsz, divide by {self.config.ppo_mini_batch_size / len(data)}, now directly devided by {self.gradient_accumulation}")
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/entropy": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/pg_clipfrac_prefix": pg_clipfrac_prefix.detach().item(),
                        "actor/pg_clipfrac_lower_prefix": pg_clipfrac_lower_prefix.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
