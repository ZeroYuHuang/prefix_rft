import hydra

import prefix_rft_v3.core_algos  # noqa: F401
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import run_ppo
from verl.utils.device import auto_set_device

from prefix_rft_v3.task_runner import PrefixRFTTaskRunner


@hydra.main(config_path="config", config_name="prefix_rft_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=PrefixRFTTaskRunner)


if __name__ == "__main__":
    main()
