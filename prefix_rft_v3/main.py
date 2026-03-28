from pathlib import Path
import shlex
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import prefix_rft_v3.core_algos  # noqa: F401
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from prefix_rft_v3.task_runner import PrefixRFTTaskRunner


def _persist_run_metadata(config) -> None:
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True), encoding="utf-8")

    overrides = OmegaConf.create(
        {
            "task_overrides": list(HydraConfig.get().overrides.task),
            "hydra_overrides": list(HydraConfig.get().overrides.hydra),
        }
    )
    (output_dir / "launch_overrides.yaml").write_text(OmegaConf.to_yaml(overrides, resolve=True), encoding="utf-8")

    launch_cmd = "python3 -m prefix_rft_v3.main"
    if len(sys.argv) > 1:
        launch_cmd = f"{launch_cmd} {shlex.join(sys.argv[1:])}"
    (output_dir / "launch_command.txt").write_text(f"{launch_cmd}\n", encoding="utf-8")


@hydra.main(config_path="config", config_name="prefix_rft_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    _persist_run_metadata(config)
    run_ppo(config, task_runner_class=PrefixRFTTaskRunner)


if __name__ == "__main__":
    main()
