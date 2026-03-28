from omegaconf import OmegaConf, open_dict

from prefix_rft_v3.core_algos import PREFIX_POLICY_LOSS_MODE
from prefix_rft_v3.dp_actor import DataParallelPPOActor
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.fsdp_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as VerlAsyncActorRolloutRefWorker


def _inject_prefix_actor_config(worker) -> None:
    prefix_cfg = worker.config.get("prefix_rft", {}) or {}

    with open_dict(worker.config.actor.policy_loss):
        worker.config.actor.policy_loss.loss_mode = prefix_cfg.get("policy_loss_mode", PREFIX_POLICY_LOSS_MODE)


def _build_prefix_actor_config(worker):
    actor_cfg = omega_conf_to_dataclass(worker.config.actor)
    actor_cfg.prefix_rft = OmegaConf.to_container(worker.config.get("prefix_rft", {}), resolve=True)
    return actor_cfg


class ActorRolloutRefWorker(VerlActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        _inject_prefix_actor_config(self)
        super().init_model()

        if self._is_actor:
            actor_cfg = _build_prefix_actor_config(self)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)


class AsyncActorRolloutRefWorker(VerlAsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        _inject_prefix_actor_config(self)
        super().init_model()

        if self._is_actor:
            actor_cfg = _build_prefix_actor_config(self)
            self.actor = DataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_ref:
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)
