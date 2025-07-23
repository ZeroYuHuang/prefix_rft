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
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from recipe.prefix_rft_v2 import core_algos
from recipe.prefix_rft_v2.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from recipe.prefix_rft_v2.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.trainer.ppo.ray_trainer import Role, WorkerType, ResourcePoolManager, reduce_metrics, _timer

WorkerType = Type[Worker]



class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PREFIX = 'grpo_prefix'
    DR_GRPO = 'dr_grpo'
    DR_GRPO_PREFIX = 'dr_grpo_prefix',
    DR_GRPO_PREFIX_V2 = 'dr_grpo_prefix_v2',

import torch
from verl.utils.torch_functional import masked_mean
from collections import defaultdict
from tensordict import TensorDict


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.DR_GRPO_PREFIX:
        advantages, returns = core_algos.compute_dr_grpo_prefix_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            prefix_mask=data.batch['prefix_mask'],
            prefix_index=data.non_tensor_batch['prefix_uid'],
            num_rollouts_per_prefix=num_repeat,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.DR_GRPO_PREFIX_V2:
        advantages, returns = core_algos.compute_dr_grpo_prefix_v2_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'],
            prefix_mask=data.batch['prefix_mask'],
            prefix_index=data.non_tensor_batch['prefix_uid'],
            num_rollouts_per_prefix=num_repeat,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.DR_GRPO:
        advantages, returns = core_algos.compute_dr_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def post_process_gen_batch(gen_batch, pad_token_id, eos_token_id, truncate_response_len):
    from verl.utils.model import compute_position_id_with_mask
    from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
    raw_prompt_lengths = list(gen_batch.non_tensor_batch["raw_prompt_lengths"])
    demo_lens = list(gen_batch.non_tensor_batch["demo_len"])
    idx = gen_batch.batch["prompts"]
    response = gen_batch.batch["responses"]

    if not torch.is_tensor(raw_prompt_lengths):
        raw_prompt_lengths = torch.tensor(raw_prompt_lengths, device=idx.device)
        demo_lens = torch.tensor(demo_lens, device=idx.device)
    batch_size, _ = idx.size()
    # 1. 计算每个样本 idx 的有效长度（即 prompt 中非 pad token 的数量，包含原始 prompt + 答案前缀）
    valid_lengths = (idx != pad_token_id).sum(dim=-1)  # shape: (batch_size,)

    # 2. 计算答案前缀长度 = 有效长度 - raw_prompt_length
    prefix_lengths = valid_lengths - raw_prompt_lengths  # 每个样本：答案前缀 token 数

    # 3. 提取每个样本的原始 prompt tokens（不包含答案前缀）
    raw_prompts_list = []
    for i in range(batch_size):
        v_len = int(valid_lengths[i].item())
        raw_len = int(raw_prompt_lengths[i].item())
        # 当有效长度等于原始 prompt长度，则没有答案前缀；否则，原始 prompt 占前 raw_len 部分
        if v_len > raw_len:
            # 从有效 tokens 中取前 raw_len 个：由于有效 tokens在右侧，所以为 idx[i, -v_len : -(v_len - raw_len)]
            raw_prompt = idx[i, -v_len: -(v_len - raw_len)]
        else:
            raw_prompt = idx[i, -v_len:]
        # 对每个样本，左填充 raw_prompt 到 batch 内最长原始 prompt 长度
        raw_prompts_list.append(raw_prompt)
    # 求本批次内每个样本 raw_prompt 的实际长度（raw_len 值均从 raw_prompt_lengths中获得）
    max_raw_prompt_len = int(raw_prompt_lengths.max().item())

    # 左 pad 每个 raw_prompt 到最大长度
    padded_raw_prompts_list = []
    for raw_prompt in raw_prompts_list:
        cur_len = raw_prompt.size(0)
        pad_size = max_raw_prompt_len - cur_len
        if pad_size > 0:
            pad_tensor = torch.full((pad_size,), pad_token_id, dtype=raw_prompt.dtype, device=raw_prompt.device)
            raw_prompt = torch.cat([pad_tensor, raw_prompt], dim=0)
        padded_raw_prompts_list.append(raw_prompt)
    # padded raw prompt tensor: (batch_size, max_raw_prompt_len)
    padded_raw_prompts = torch.stack(padded_raw_prompts_list, dim=0)

    # 4. 构造完整回答：答案前缀 + 模型续写 tokens
    full_responses_list = []
    for i in range(batch_size):
        v_len = int(valid_lengths[i].item())
        p_len = int(prefix_lengths[i].item())
        # 提取答案前缀（在有效 tokens的末尾部分），如果 p_len 为 0 则为空
        if p_len > 0:
            answer_prefix = idx[i, -p_len:]
        else:
            answer_prefix = torch.tensor([], dtype=idx.dtype, device=idx.device)
        
        # 提取对应样本的模型生成续写 tokens
        # 如果 response 为列表或元组，则直接取 response[i]；若为 tensor，则 response[i] 即为该样本续写
        if isinstance(response, (list, tuple)):
            gen_cont = response[i]
        else:
            gen_cont = response[i]
        
        # 拼接得到完整回答 tokens
        full_response = torch.cat([answer_prefix, gen_cont], dim=0)[:truncate_response_len]
        # if full_response.size(0) >= truncate_response_len:
        #     print(f"The full response is: {full_response.size(0)}, truncated to: {truncate_response_len}, the answer prefix length is: {answer_prefix.size(0)}")
        #     full_response = full_response[:truncate_response_len]
        full_responses_list.append(full_response)

    # 5. 对完整回答进行右侧 padding：对每个样本的 full_response 右 pad 至本批最大长度
    
    response_lengths = [fr.size(0) for fr in full_responses_list]
    max_response_len = max(response_lengths)

    # padded_responses_list = []
    # for fr in full_responses_list:
    #     cur_len = fr.size(0)
    #     pad_len = max_response_len - cur_len
    #     if pad_len > 0:
    #         pad_tensor = torch.full((pad_len,), pad_token_id, dtype=fr.dtype, device=fr.device)
    #         fr = torch.cat([fr, pad_tensor], dim=0)  # 右 pad
    #     padded_responses_list.append(fr)
    # # 合并所有样本，得到 padded_responses: (batch_size, max_response_len)
    # padded_responses = torch.stack(padded_responses_list, dim=0)

    padded_responses = pad_2d_list_to_length(full_responses_list, pad_token_id,
                                             max_length=max_response_len).to(idx.device)   


    # 6. 构造新的 attention_mask
    #    对于 raw_prompts（左 padding），mask：非 pad token为1，pad token为0
    prompt_attention_mask = (padded_raw_prompts != pad_token_id).long()
    #    对于 responses（右 padding），mask：非 pad token为1，pad token为0
    # response_attention_mask = (padded_responses != pad_token_id).long()
    response_attention_mask = get_response_mask(response_id=padded_responses,
                                                eos_token=eos_token_id,
                                                dtype=prompt_attention_mask.dtype)
    # 拼接 prompt 与 response 得到整个序列的 mask
    final_attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=-1)

    # 7. 构造新的 position_ids
    # 对于 prompt 部分：
    prompt_position_ids = compute_position_id_with_mask(prompt_attention_mask)
    # 对于 response 部分：每个样本的 position ids 从 prompt 最后一个有效 token位置+1 开始
    response_length = padded_responses.size(1)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
    response_position_ids = torch.zeros_like(padded_responses)
    response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
    # for i in range(batch_size):
    #     raw_len = int(raw_prompt_lengths[i].item())
    #     start_pos = raw_len if raw_len > 0 else 0
    #     pos = torch.arange(start_pos, start_pos + max_response_len, device=idx.device)
    #     response_position_ids[i] = pos
    # 拼接 prompt 与 response 部分的 position_ids
    final_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    
    # 8. 构造最终 input_ids = prompts + responses（prompts指经过左 padding 的 raw_prompts）
    final_input_ids = torch.cat([padded_raw_prompts, padded_responses], dim=-1)

    # 9. 构造 prefix_mask：
    # 对于每个样本，其 response 部分有 max_response_len 个位置，
    # 我们根据 prefix_lengths 得到相应的 mask：前 prefix_lengths[i] 个位置标记为1，其余位置标记为0
    # 注意：prefix_lengths 为 1D Tensor，故需要 unsqueeze 到 (batch_size, 1)
    prefix_mask = (torch.arange(max_response_len, device=idx.device)
                   .unsqueeze(0).expand(batch_size, max_response_len)
                   < prefix_lengths.unsqueeze(1)).long()

    # 10. 计算 relative_prefix_position
    # pos_idx: 1...max_response_len
    pos_idx = torch.arange(1, max_response_len + 1, device=idx.device).unsqueeze(0).expand(batch_size, -1)
    # demo_lens -> (B,1)，对 0 做 clamp，使其至少为 1
    demo_lens_unsq = demo_lens.unsqueeze(1)                  # (B,1)
    safe_demo_lens = demo_lens_unsq.clamp(min=1)             # 把 0 -> 1，避免除零
    # 先按 pos_idx / safe_demo_lens 计算
    relpos = pos_idx.float() / safe_demo_lens.float()        # (B, L_resp)
    # 对「非 prefix 且非 pad」的位置直接置 1
    non_prefix_and_non_pad = (prefix_mask == 0) & (response_attention_mask == 1)
    relpos[non_prefix_and_non_pad] = -1.0
    # 对 pad 部分保持 0
    prefix_relpos = relpos * response_attention_mask.float()       # (B, L_resp)

    
    # 11. 封装成最终 batch，注意 TensorDict 的实现可能因使用不同框架而不同，
    #    这里假设 TensorDict 可以接收一个字典及 batch_size 参数构造
    batch = TensorDict(
        {
            'prompts': padded_raw_prompts,      # shape: (batch_size, max_raw_prompt_len)
            'responses': padded_responses,        # shape: (batch_size, max_response_len)
            'input_ids': final_input_ids,         # shape: (batch_size, max_raw_prompt_len + max_response_len)
            'attention_mask': final_attention_mask,
            'position_ids': final_position_ids,
            'prefix_mask': prefix_mask,
            'prefix_relpos': prefix_relpos,
        },
        batch_size=batch_size
    )
    
    return DataProto(batch=batch, non_tensor_batch=gen_batch.non_tensor_batch)


class RayTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.DR_GRPO, 
                AdvantageEstimator.DR_GRPO_PREFIX,
                AdvantageEstimator.DR_GRPO_PREFIX_V2, 
                AdvantageEstimator.GRPO, 
                AdvantageEstimator.GRPO_PREFIX,
                AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        os.makedirs(self.config.trainer.default_prefix_log_dir, exist_ok=True)
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def save_prefix_log(self, batch, global_step):
        """
        print(batch.batch.keys())
        print(batch.non_tensor_batch.keys())
        _StringKeys(dict_keys(['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids', 'prefix_mask', 'response_mask', 'old_entropy', 'old_log_probs', 'token_level_scores', 'token_level_rewards', 'advantages', 'returns']))
        dict_keys(['data_source', 'target', 'ability', 'reward_model', 'extra_info', 'demos', 'avg_rewards', 'prefix_ratios', 'uid', 'raw_prompt_lengths', 'index', 'prefix_ratio', 'prefix_uid', 'prefix_correct'])
        """
        import csv
        data = []
        data.append(["uid", "prefix_uid", "prefix_ratio", "prefix_correct", "reward", "advantages"])
        batch_size = batch.batch.batch_size[0]
        prefix_uid = batch.non_tensor_batch["prefix_uid"]

        for i in range(batch_size):
            if prefix_uid[i] != '0000-0000-0000-0000': # is non_zero_prefix
                _data = [
                    batch.non_tensor_batch["uid"][i],
                    prefix_uid[i],
                    batch.non_tensor_batch["prefix_ratio"][i],
                    batch.non_tensor_batch["prefix_correct"][i],
                    batch.batch['token_level_rewards'][i].sum().numpy(),
                    batch.batch['advantages'][i, 0].numpy()
                ]
                data.append(_data)
        
        # 写入 CSV 文件
        with open(f'{self.config.trainer.default_prefix_log_dir}/{global_step}.csv', mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
            "dr_grpo"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from recipe.prefix_rft_v2.main import create_perfix_rft_dataset, create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_perfix_rft_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            reverse_indices = self._reorder_batch_randomly(test_gen_batch_padded)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()
            test_output_gen_batch_padded.reorder(reverse_indices)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _reorder_batch_randomly(self, batch: DataProto):
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        random_indices = torch.randperm(batch_size)
        inverse_indices = torch.argsort(random_indices)
        batch.reorder(random_indices)
        return inverse_indices

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.train_dataloader.dataset.global_step = self.global_steps

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_repeated = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):

                        gen_batch_repeated = []
                        for i in range(self.config.data.num_prefix):
                            batch.pop(
                                batch_keys=[f'input_ids_{i}', f'attention_mask_{i}', f'position_ids_{i}'],
                                non_tensor_batch_keys=[f'prefix_ratio_{i}', f'prefix_uid_{i}', f'prefix_correct_{i}', f"demo_len_{i}", f"prefix_ctrl_low_{i}", f"prefix_ctrl_high_{i}"]
                            )
                            gen_batch_repeated_i = batch_repeated.pop(batch_keys=[f'input_ids_{i}', f'attention_mask_{i}', f'position_ids_{i}'])
                            gen_batch_repeated_i.batch["input_ids"] = gen_batch_repeated_i.batch[f'input_ids_{i}']
                            gen_batch_repeated_i.batch["attention_mask"] = gen_batch_repeated_i.batch[f'attention_mask_{i}']
                            gen_batch_repeated_i.batch["position_ids"] = gen_batch_repeated_i.batch[f'position_ids_{i}']
                            gen_batch_repeated_i.non_tensor_batch["prefix_ctrl_low"] = batch_repeated.non_tensor_batch[f"prefix_ctrl_low_{i}"]
                            gen_batch_repeated_i.non_tensor_batch["prefix_ctrl_high"] = batch_repeated.non_tensor_batch[f"prefix_ctrl_high_{i}"]
                            gen_batch_repeated_i.non_tensor_batch["raw_prompt_lengths"] = batch_repeated.non_tensor_batch['raw_prompt_lengths']
                            gen_batch_repeated_i.non_tensor_batch["prefix_ratio"] = batch_repeated.non_tensor_batch[f"prefix_ratio_{i}"]
                            gen_batch_repeated_i.non_tensor_batch["prefix_uid"] = batch_repeated.non_tensor_batch[f'prefix_uid_{i}']
                            gen_batch_repeated_i.non_tensor_batch["prefix_correct"] = batch_repeated.non_tensor_batch[f'prefix_correct_{i}']
                            gen_batch_repeated_i.non_tensor_batch["demo_len"] = batch_repeated.non_tensor_batch[f"demo_len_{i}"]
                            gen_batch_repeated_i.pop(
                                batch_keys=[f'input_ids_{i}', f'attention_mask_{i}', f'position_ids_{i}'],
                                # non_tensor_batch_keys=['raw_prompt_ids', 'raw_prompt_lengths'],
                            )
                            gen_batch_repeated.append(gen_batch_repeated_i)

                        # concat and reorder
                        gen_batch_repeated = DataProto.concat(gen_batch_repeated)
                        indices = []
                        for j in range(self.config.data.train_batch_size):
                            for k in range(self.config.data.num_prefix):
                                indices += [
                                    j* self.config.actor_rollout_ref.rollout.n + l + k * self.config.actor_rollout_ref.rollout.n * self.config.data.train_batch_size
                                    for l in range(self.config.actor_rollout_ref.rollout.n)
                                ]
                        indices = torch.tensor(indices)
                        gen_batch_repeated.reorder(torch.tensor(indices))
                        
                        # for balanced rollout
                        reverse_indices = self._reorder_batch_randomly(gen_batch_repeated)
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_repeated)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_repeated)
                            self.async_rollout_manager.sleep()
                        gen_batch_output.reorder(reverse_indices)

                        gen_batch_output = post_process_gen_batch(
                            gen_batch_output, 
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            # truncate_response_len=self.config.data.max_response_length
                            truncate_response_len=self.config.actor_rollout_ref.rollout.truncate_length
                        )
                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     with _timer('gen_max', timing_raw):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info['do_sample'] = False
                    #         gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                    #         batch = batch.union(gen_baseline_output)
                    #         reward_baseline_tensor = self.reward_fn(batch)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    #         batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    #         batch.batch['reward_baselines'] = reward_baseline_tensor

                    #         del gen_baseline_batch, gen_baseline_output

                    # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n * self.config.data.num_prefix, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # print("The demonstration lengths are:")
                    # print(batch.non_tensor_batch["demo_len"][batch.non_tensor_batch["prefix_uid"] != '0000-0000-0000-0000'])
                    # prefix related metrics
                    prefix_metrics = {
                        "actor/prefix_low": batch.non_tensor_batch["prefix_ctrl_low"][batch.non_tensor_batch["prefix_uid"] != '0000-0000-0000-0000'].mean(),
                        "actor/prefix_high": batch.non_tensor_batch["prefix_ctrl_high"][batch.non_tensor_batch["prefix_uid"] != '0000-0000-0000-0000'].mean(),
                        "actor/prefix_ratio": batch.non_tensor_batch["prefix_ratio"][batch.non_tensor_batch["prefix_uid"] != '0000-0000-0000-0000'].mean(),
                        "actor/demo_len": batch.non_tensor_batch["demo_len"][batch.non_tensor_batch["prefix_uid"] != '0000-0000-0000-0000'].mean()
                    }
                    metrics.update(prefix_metrics)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # compute prefix ctrl out
                    prefix_ctrl_low, prefix_ctrl_high = batch.non_tensor_batch["prefix_ctrl_low"], batch.non_tensor_batch["prefix_ctrl_high"]
                    prefix_ctrl_low, prefix_ctrl_high = np.mean(prefix_ctrl_low[prefix_ctrl_low != -1]), np.mean(prefix_ctrl_high[prefix_ctrl_high != -1])
                    batch.meta_info["prefix_ctrl_low"] = prefix_ctrl_low
                    batch.meta_info["prefix_ctrl_high"] = prefix_ctrl_high

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn, cur_step=self.global_steps)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode="token-mean")
                        old_log_prob_metrics = {
                            "actor/entropy_loss": entropy_loss.detach().item(),
                            "actor/num_response_tokens": torch.sum(response_masks).detach().item(),
                            "actor/num_prefix_tokens": torch.sum(batch.batch["prefix_mask"]).detach().item(),
                            "actor/off_ratio": torch.sum(batch.batch["prefix_mask"]).detach().item() / torch.sum(response_masks).detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        # old_log_prob.batch['old_entropy'] = entropys.cpu()
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )
                        # TODO 储存下来和Prefix有关的statistic
                        print(batch.batch.keys())
                        print(batch.non_tensor_batch.keys())
                        self.save_prefix_log(batch=batch, global_step=self.global_steps)
                        batch.meta_info["global_step"] = self.global_steps

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.train_dataloader.dataset.global_step = self.global_steps