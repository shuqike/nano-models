"""main.py
Hydra reads YAMLs into an OmegaConf DictConfig which is typed and supports interpolation.
Example: `python -m mlproj.main data=cifar10 model=resnet18 optim.lr=3e-4 trainer.max_epochs=20`
"""
import os
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
from coolname import generate_slug
import wandb

from .schemas import Config
from .train import train


def dictconfig_to_pydantic(cfg: DictConfig) -> Config:
    # resolve interpolations, convert to primitive dict
    payload = OmegaConf.to_container(cfg, resolve=True)
    return Config.model_validate(payload)  # pydantic v2


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Pretty print the composed config
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Generate a name if missing
    if cfg.run.name in (None, "null"):
        slug = generate_slug(2)  # e.g., "silent-lion-7"
        cfg.run.name = slug

    # Validate with Pydantic
    pcfg = dictconfig_to_pydantic(cfg)

    # Initialize Weights & Biases
    if pcfg.wandb.enable:
        run = wandb.init(
            project=pcfg.wandb.project,
            entity=pcfg.wandb.entity,
            name=pcfg.run.name,
            group=pcfg.run.group,
            tags=pcfg.wandb.tags,
            notes=pcfg.wandb.notes,
            config=pcfg.model_dump(),  # log validated config
        )
    else:
        run = None

    # Train
    metrics = train(cfg, pcfg, wandb_run=run)

    # Log final metrics and finish W&B
    if run is not None:
        wandb.log({f"final/{k}": v for k, v in metrics.items()})
        run.finish()


if __name__ == "__main__":
    main()
