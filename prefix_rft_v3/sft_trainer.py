import hydra
from omegaconf import open_dict

from prefix_rft_v3.templates import Llama_3p1_8B_chat_template, Qwen3_8B_Base_chat_template
from verl.trainer.sft_trainer import run_sft
from verl.utils.device import auto_set_device


def _maybe_get_custom_chat_template(model_path: str) -> str | None:
    if "Llama-3.1-8B" in model_path:
        return Llama_3p1_8B_chat_template
    if "Qwen3" in model_path:
        return Qwen3_8B_Base_chat_template
    return None


@hydra.main(config_path="../verl/trainer/config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    auto_set_device(config)

    custom_chat_template = _maybe_get_custom_chat_template(config.model.path)
    if custom_chat_template is not None:
        with open_dict(config.model):
            config.model.custom_chat_template = custom_chat_template

    run_sft(config)


if __name__ == "__main__":
    main()
