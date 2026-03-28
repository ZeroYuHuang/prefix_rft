import logging
import os

import torch

from prefix_rft_v3.core_algos import PREFIX_POLICY_LOSS_MODE
from prefix_rft_v3.scheduler.global_step import CTRL_MAPPING
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor as VerlDataParallelPPOActor

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(VerlDataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)

        prefix_cfg = getattr(config, "prefix_rft", {}) or {}
        self.prefix_rft_config = dict(prefix_cfg)

        reshapers = self.prefix_rft_config.get("off_adv_reshaper", "identity")
        if isinstance(reshapers, str):
            self.reshapers = [item.strip() for item in reshapers.split(",") if item.strip()]
        elif reshapers:
            self.reshapers = [str(item) for item in reshapers]
        else:
            self.reshapers = ["identity"]

        self.entropy_mode = self.prefix_rft_config.get("entropy_mode", "response_only")
        self.replace_old_log_prob = bool(self.prefix_rft_config.get("replace_old_log_prob", False))
        self.requires_entropy_reshape = any(
            name in {"entropy", "entropy_low", "entropy_op"} for name in self.reshapers
        )

        self.ent_mask_ratio_ctrl = None
        if any(name in {"entropy", "entropy_low", "entropy_op", "random"} for name in self.reshapers):
            ctrl_type = self.prefix_rft_config.get("off_ent_mask_ratio_ctrl_type", "const")
            ctrl_kwargs = dict(self.prefix_rft_config.get("off_ent_mask_ratio_ctrl", {}).get("kwargs", {}))
            self.ent_mask_ratio_ctrl = CTRL_MAPPING[ctrl_type](**ctrl_kwargs)

    def _mask_low_ranked_tokens(self, adv, token_mask, ranking_values, mask_ratio):
        token_mask = token_mask.bool()
        if not token_mask.any():
            return adv

        selected_positions = token_mask.nonzero(as_tuple=False)
        selected_values = ranking_values[token_mask]
        if selected_values.numel() == 0:
            return adv

        split_point = int(selected_values.numel() * mask_ratio)
        if split_point <= 0:
            return adv

        _, ranked_indices = torch.sort(selected_values)
        masked_positions = selected_positions[ranked_indices[:split_point]]
        adv[masked_positions[:, 0], masked_positions[:, 1]] = 0
        return adv

    def reshape_func(self, adv, log_prob, prefix_mask, **kwargs):
        device = adv.device
        prefix_mask = prefix_mask.to(device=device).bool()
        if not prefix_mask.any():
            return adv, log_prob

        global_step = kwargs.get("global_step", 0)

        def identity(cur_adv, cur_log_prob):
            return cur_adv, cur_log_prob

        def importance_sampling(cur_adv, cur_log_prob):
            old_log_prob = kwargs.get("old_log_prob")
            if old_log_prob is None:
                return cur_adv, cur_log_prob
            old_log_prob = old_log_prob.to(device)
            cur_log_prob[prefix_mask] = cur_log_prob[prefix_mask] + old_log_prob[prefix_mask]
            return cur_adv, cur_log_prob

        def luffy(cur_adv, cur_log_prob):
            old_log_prob = kwargs.get("old_log_prob")
            if old_log_prob is None:
                return cur_adv, cur_log_prob
            old_log_prob = old_log_prob.to(device)
            cur_log_prob[prefix_mask] = (
                cur_log_prob[prefix_mask]
                + old_log_prob[prefix_mask]
                - torch.log(torch.exp(cur_log_prob[prefix_mask]) + 0.1)
            )
            return cur_adv, cur_log_prob

        def silu(cur_adv, cur_log_prob):
            cur_adv[prefix_mask] = torch.nn.functional.silu(cur_adv[prefix_mask])
            return cur_adv, cur_log_prob

        def relu(cur_adv, cur_log_prob):
            cur_adv[prefix_mask] = torch.nn.functional.relu(cur_adv[prefix_mask])
            return cur_adv, cur_log_prob

        def const(cur_adv, cur_log_prob):
            cur_adv[prefix_mask] = float(os.getenv("CONST_ADV_MASK_VAL", 0.001))
            return cur_adv, cur_log_prob

        def entropy_op(cur_adv, cur_log_prob):
            entropy = kwargs.get("entropy")
            response_mask = kwargs.get("response_mask")
            if entropy is None or response_mask is None:
                return cur_adv, cur_log_prob
            entropy = entropy.to(device)
            response_mask = response_mask.to(device=device).bool()
            on_policy_mask = (~prefix_mask) & response_mask
            ratio = 0.8 if self.ent_mask_ratio_ctrl is None else self.ent_mask_ratio_ctrl.value(global_step=global_step)
            cur_adv = self._mask_low_ranked_tokens(cur_adv, on_policy_mask, entropy, ratio)
            return cur_adv, cur_log_prob

        def entropy(cur_adv, cur_log_prob):
            entropy_tensor = kwargs.get("entropy")
            if entropy_tensor is None:
                return cur_adv, cur_log_prob
            entropy_tensor = entropy_tensor.to(device)
            ratio = 0.8 if self.ent_mask_ratio_ctrl is None else self.ent_mask_ratio_ctrl.value(global_step=global_step)
            cur_adv = self._mask_low_ranked_tokens(cur_adv, prefix_mask, entropy_tensor, ratio)
            return cur_adv, cur_log_prob

        def entropy_low(cur_adv, cur_log_prob):
            return entropy(cur_adv, cur_log_prob)

        def position(cur_adv, cur_log_prob, eps=1e-3):
            prefix_relpos = kwargs.get("prefix_relpos")
            a = kwargs.get("prefix_ctrl_low")
            b = kwargs.get("prefix_ctrl_high")
            if prefix_relpos is None or a is None or b is None or a < 0 or b < 0 or a == b:
                return cur_adv, cur_log_prob

            prefix_relpos = prefix_relpos.to(device)
            denom = max(float(b) ** 2 - float(a) ** 2, eps)
            f = torch.where(
                prefix_relpos <= float(a),
                torch.full_like(prefix_relpos, 2.0 / (float(a) + float(b))),
                torch.where(
                    prefix_relpos <= float(b),
                    2.0 * (float(b) - prefix_relpos) / denom,
                    torch.zeros_like(prefix_relpos),
                ),
            )
            valid_mask = prefix_mask & (prefix_relpos >= 0)
            if not valid_mask.any():
                return cur_adv, cur_log_prob
            w = 1.0 / f.clamp_min(eps)
            mean_w = w[valid_mask].mean()
            cur_adv[valid_mask] = (w[valid_mask] / mean_w) * cur_adv[valid_mask]
            return cur_adv, cur_log_prob

        def random_mask(cur_adv, cur_log_prob):
            ratio = 0.8 if self.ent_mask_ratio_ctrl is None else self.ent_mask_ratio_ctrl.value(global_step=global_step)
            selected_positions = prefix_mask.nonzero(as_tuple=False)
            if selected_positions.numel() == 0:
                return cur_adv, cur_log_prob
            num_to_mask = int(selected_positions.size(0) * ratio)
            if num_to_mask <= 0:
                return cur_adv, cur_log_prob
            shuffled = selected_positions[torch.randperm(selected_positions.size(0), device=device)[:num_to_mask]]
            cur_adv[shuffled[:, 0], shuffled[:, 1]] = 0
            return cur_adv, cur_log_prob

        def length_on_pos(cur_adv, cur_log_prob):
            response_mask = kwargs.get("response_mask")
            if response_mask is None:
                return cur_adv, cur_log_prob
            response_mask = response_mask.to(device)
            max_seq_length = response_mask.shape[-1]
            response_length = response_mask.sum(dim=-1)
            end_hit_mask = (response_length == max_seq_length).float()
            on_policy_mask = ((~prefix_mask) & response_mask.bool()).float()
            zero_adv_mask = end_hit_mask.unsqueeze(-1) * on_policy_mask * (cur_adv > 0).float()
            cur_adv = cur_adv * (1 - zero_adv_mask)
            return cur_adv, cur_log_prob

        def length_all_pos(cur_adv, cur_log_prob):
            response_mask = kwargs.get("response_mask")
            if response_mask is None:
                return cur_adv, cur_log_prob
            response_mask = response_mask.to(device)
            max_seq_length = response_mask.shape[-1]
            response_length = response_mask.sum(dim=-1)
            end_hit_mask = (response_length == max_seq_length).float()
            zero_adv_mask = end_hit_mask.unsqueeze(-1) * response_mask * (cur_adv > 0).float()
            cur_adv = cur_adv * (1 - zero_adv_mask)
            return cur_adv, cur_log_prob

        def length(cur_adv, cur_log_prob):
            response_mask = kwargs.get("response_mask")
            if response_mask is None:
                return cur_adv, cur_log_prob
            response_mask = response_mask.to(device)
            max_seq_length = response_mask.shape[-1]
            response_length = response_mask.sum(dim=-1)
            end_hit_mask = (response_length == max_seq_length).float()
            zero_adv_mask = end_hit_mask.unsqueeze(-1) * response_mask
            cur_adv = cur_adv * (1 - zero_adv_mask)
            return cur_adv, cur_log_prob

        reshape_fn_mapping = {
            "identity": identity,
            "importance_sampling": importance_sampling,
            "luffy": luffy,
            "silu": silu,
            "relu": relu,
            "const": const,
            "entropy": entropy,
            "entropy_low": entropy_low,
            "entropy_op": entropy_op,
            "position": position,
            "random": random_mask,
            "length_on_pos": length_on_pos,
            "length_all_pos": length_all_pos,
            "length": length,
        }

        for reshaper in self.reshapers:
            reshape_fn = reshape_fn_mapping.get(reshaper)
            if reshape_fn is not None:
                adv, log_prob = reshape_fn(adv, log_prob)

        return adv, log_prob

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        global_step = data.meta_info.get("global_step", data.meta_info.get("global_steps", 0))
        prefix_ctrl_low = data.meta_info.get("prefix_ctrl_low", -1.0)
        prefix_ctrl_high = data.meta_info.get("prefix_ctrl_high", -1.0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        has_prefix_data = "prefix_mask" in data.batch.keys()
        if has_prefix_data:
            select_keys.append("prefix_mask")
            if "prefix_relpos" in data.batch.keys():
                select_keys.append("prefix_relpos")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}

                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    prefix_mask = model_inputs.get("prefix_mask")
                    prefix_relpos = model_inputs.get("prefix_relpos")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = (
                        self.config.calculate_entropy or (entropy_coeff != 0) or self.requires_entropy_reshape
                    )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()

                    if prefix_mask is not None:
                        advantages, log_prob = self.reshape_func(
                            adv=advantages,
                            log_prob=log_prob,
                            prefix_mask=prefix_mask,
                            old_log_prob=old_log_prob,
                            entropy=entropy.detach() if entropy is not None else None,
                            global_step=global_step,
                            prefix_relpos=prefix_relpos,
                            prefix_ctrl_low=prefix_ctrl_low,
                            prefix_ctrl_high=prefix_ctrl_high,
                            response_mask=response_mask,
                        )
                        if self.replace_old_log_prob:
                            old_log_prob = old_log_prob.clone()
                            old_log_prob[prefix_mask.bool()] = log_prob[prefix_mask.bool()].detach()

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    policy_loss_kwargs = {
                        "old_log_prob": old_log_prob,
                        "log_prob": log_prob,
                        "advantages": advantages,
                        "response_mask": response_mask,
                        "loss_agg_mode": loss_agg_mode,
                        "config": self.config,
                        "rollout_is_weights": rollout_is_weights,
                    }
                    if loss_mode == PREFIX_POLICY_LOSS_MODE:
                        policy_loss_kwargs["prefix_mask"] = prefix_mask
                    pg_loss, pg_metrics = policy_loss_fn(**policy_loss_kwargs)
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_mask = response_mask
                        if prefix_mask is not None and self.entropy_mode == "response_only":
                            entropy_mask = response_mask * (~prefix_mask.bool()).to(response_mask.dtype)
                        entropy_agg = agg_loss(
                            loss_mat=entropy,
                            loss_mask=entropy_mask,
                            loss_agg_mode=loss_agg_mode,
                            loss_scale_factor=self.config.loss_scale_factor,
                        )
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld,
                            loss_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            loss_scale_factor=self.config.loss_scale_factor,
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
