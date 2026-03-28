import os
import socket

import ray

from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config

from prefix_rft_v3.templates import Llama_3p1_8B_chat_template, Qwen3_8B_Base_chat_template


def _maybe_get_custom_chat_template(model_path: str) -> str | None:
    if "Llama-3.1-8B" in model_path:
        return Llama_3p1_8B_chat_template
    if "Qwen3" in model_path:
        return Qwen3_8B_Base_chat_template
    return None


def _copy_config_value(value, omega_conf):
    if omega_conf.is_config(value) or isinstance(value, (dict, list)):
        return omega_conf.create(omega_conf.to_container(value, resolve=True))
    return value


@ray.remote(num_cpus=1)
class PrefixRFTTaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        use_prefix_worker = config.get("prefix_rft", {}).get("enable", False)

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            if use_prefix_worker:
                from prefix_rft_v3.fsdp_workers import AsyncActorRolloutRefWorker
            else:
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        elif config.critic.strategy in {"veomni", "torchtitan"}:
            if use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker

                CriticWorker = TrainingWorker
            else:
                raise ValueError(
                    f"Invalid use_legacy_worker_impl for {config.critic.strategy}: {use_legacy_worker_impl}"
                )
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def add_reward_model_resource_pool(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward.reward_model.enable:
            if config.reward.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
                self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return

        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")
            resource_pool_spec["reward_pool"] = [
                config.reward.reward_model.n_gpus_per_node
            ] * config.reward.reward_model.nnodes
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        custom_chat_template = _maybe_get_custom_chat_template(config.actor_rollout_ref.model.path)
        if custom_chat_template is not None:
            with open_dict(config.actor_rollout_ref.model):
                config.actor_rollout_ref.model.custom_chat_template = custom_chat_template

        if config.get("prefix_rft", {}).get("enable", False):
            prefix_cfg = config.get("prefix_rft", {})
            with open_dict(config.data):
                if "custom_cls" not in config.data or config.data.custom_cls is None:
                    config.data.custom_cls = {}
                config.data.custom_cls.path = "pkg://prefix_rft_v3.rl_dataset"
                config.data.custom_cls.name = "PrefixRFTDataset"
                config.data.return_raw_chat = True

                prefix_defaults = {
                    "demo_ratio": prefix_cfg.get("demo_ratio", 1.0),
                    "demo_key": prefix_cfg.get("demo_key", "demos"),
                    "demo_corr_key": prefix_cfg.get("demo_corr_key", "demos_corr"),
                    "total_demo_n": prefix_cfg.get("total_demo_n", 1),
                    "num_prefix": prefix_cfg.get("num_prefix", 1),
                    "num_empty_prefix": prefix_cfg.get("num_empty_prefix", 0),
                    "min_prefix_len": prefix_cfg.get("min_prefix_len", 32),
                    "only_keep_dp_with_demo": prefix_cfg.get("only_keep_dp_with_demo", False),
                    "prefix_low_ctrl_type": prefix_cfg.get("prefix_low_ctrl_type", "const"),
                    "prefix_low_ctrl": prefix_cfg.get("prefix_low_ctrl", {"kwargs": {"init": 0.2}}),
                    "prefix_high_ctrl_type": prefix_cfg.get("prefix_high_ctrl_type", "const"),
                    "prefix_high_ctrl": prefix_cfg.get("prefix_high_ctrl", {"kwargs": {"init": 0.8}}),
                    "prefix_sampler": prefix_cfg.get("prefix_sampler", {"kwargs": {"alpha": 1.0, "beta": 1.0}}),
                }
                for key, value in prefix_defaults.items():
                    if key not in config.data or config.data.get(key) is None:
                        config.data[key] = _copy_config_value(value, OmegaConf)

                optional_prefix_fields = (
                    "prefix_low_ctrl_wrapper_type",
                    "prefix_low_ctrl_wrapper",
                    "prefix_high_ctrl_wrapper_type",
                    "prefix_high_ctrl_wrapper",
                )
                for key in optional_prefix_fields:
                    value = prefix_cfg.get(key, None)
                    if value is not None and (key not in config.data or config.data.get(key) is None):
                        config.data[key] = _copy_config_value(value, OmegaConf)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        if custom_chat_template is not None:
            if processor is not None:
                processor.chat_template = custom_chat_template
            else:
                tokenizer.chat_template = custom_chat_template

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if hasattr(train_dataset, "is_train"):
            train_dataset.is_train = True
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        if hasattr(val_dataset, "is_train"):
            val_dataset.is_train = False
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer_cls = None
        if config.get("prefix_rft", {}).get("enable", False):
            from prefix_rft_v3.ray_trainer import RayPPOTrainer as PrefixRayPPOTrainer

            trainer_cls = PrefixRayPPOTrainer
        else:
            from verl.trainer.ppo.ray_trainer import RayPPOTrainer as VerlRayPPOTrainer

            trainer_cls = VerlRayPPOTrainer

        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()
