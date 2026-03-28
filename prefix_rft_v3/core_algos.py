from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig

PREFIX_POLICY_LOSS_MODE = "prefix_v2"


@register_policy_loss(PREFIX_POLICY_LOSS_MODE)
def compute_policy_loss_prefix_v2(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert config is not None
    assert not isinstance(config, AlgoConfig)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get("clip_ratio_c", 3.0)

    prefix_cfg = getattr(config, "prefix_rft", {}) or {}
    clip_ratio_low_off = prefix_cfg.get("clip_ratio_low_off", None)
    clip_ratio_high_off = prefix_cfg.get("clip_ratio_high_off", None)
    enable_clip = bool(prefix_cfg.get("enable_clip", True))
    if clip_ratio_low_off is None:
        clip_ratio_low_off = clip_ratio_low
    if clip_ratio_high_off is None:
        clip_ratio_high_off = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    pg_clipfrac_prefix = torch.tensor(0.0, device=log_prob.device)
    pg_clipfrac_lower_prefix = torch.tensor(0.0, device=log_prob.device)
    if prefix_mask is not None:
        prefix_mask = prefix_mask.to(device=log_prob.device).bool()
        if prefix_mask.any():
            prefix_loss_mask = prefix_mask.to(response_mask.dtype)
            off_pg_losses = pg_losses1
            if enable_clip:
                off_pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low_off, 1 + clip_ratio_high_off)
                off_clip_pg_losses1 = torch.maximum(off_pg_losses, off_pg_losses2)
                pg_clipfrac_prefix = verl_F.masked_mean(
                    torch.gt(off_pg_losses2, off_pg_losses).float(), prefix_loss_mask
                )

                off_pg_losses3 = -advantages * clip_ratio_c
                off_clip_pg_losses2 = torch.min(off_pg_losses3, off_clip_pg_losses1)
                pg_clipfrac_lower_prefix = verl_F.masked_mean(
                    torch.gt(off_clip_pg_losses1, off_pg_losses3) * (advantages < 0).float(),
                    prefix_loss_mask,
                )
                off_pg_losses = torch.where(advantages < 0, off_clip_pg_losses2, off_clip_pg_losses1)
            pg_losses = torch.where(prefix_mask, off_pg_losses, pg_losses)

    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/pg_clipfrac_prefix": pg_clipfrac_prefix.detach().item(),
        "actor/pg_clipfrac_lower_prefix": pg_clipfrac_lower_prefix.detach().item(),
    }
    return pg_loss, pg_metrics
