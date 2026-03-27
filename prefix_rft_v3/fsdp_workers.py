from omegaconf import OmegaConf, open_dict

from prefix_rft_v3.dp_actor import DataParallelPPOActor
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as VerlAsyncActorRolloutRefWorker


def _inject_prefix_actor_config(worker) -> None:
    prefix_cfg = worker.config.get("prefix_rft", {}) or {}

    actor_defaults = {
        "off_adv_reshaper": prefix_cfg.get("off_adv_reshaper", "entropy"),
        "off_ent_mask_ratio_ctrl_type": prefix_cfg.get("off_ent_mask_ratio_ctrl_type", "const"),
        "off_ent_mask_ratio_ctrl": prefix_cfg.get("off_ent_mask_ratio_ctrl", {"kwargs": {"init": 0.8}}),
        "entropy_mode": prefix_cfg.get("entropy_mode", "response_only"),
        "replace_old_log_prob": prefix_cfg.get("replace_old_log_prob", False),
    }

    with open_dict(worker.config.actor):
        worker.config.actor.prefix_rft = OmegaConf.create(OmegaConf.to_container(prefix_cfg, resolve=True))
        for key, value in actor_defaults.items():
            worker.config.actor[key] = OmegaConf.create(value) if isinstance(value, dict) else value

    if hasattr(worker.config, "ref"):
        with open_dict(worker.config.ref):
            worker.config.ref.prefix_rft = OmegaConf.create(OmegaConf.to_container(prefix_cfg, resolve=True))


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    def init_model(self):
        _inject_prefix_actor_config(self)
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)


class AsyncActorRolloutRefWorker(VerlAsyncActorRolloutRefWorker):
    def init_model(self):
        _inject_prefix_actor_config(self)
        super().init_model()

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
