"""mlproj/schemas.py
Pydantic gives you validation and auto-docs.
"""
from pydantic import BaseModel, field_validator
from typing import Literal, Optional


class DataCfg(BaseModel):
    name: Literal["mnist", "cifar10"]
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4


class ModelCfg(BaseModel):
    name: Literal["mlp", "resnet18"]
    hidden_dim: int = 256
    num_classes: int = 10


class OptimCfg(BaseModel):
    name: Literal["sgd", "adam"] = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0

    @field_validator("lr")
    @classmethod
    def check_lr(cls, v):
        assert v > 0, "lr must be positive"
        return v


class TrainerCfg(BaseModel):
    max_epochs: int = 5
    device: Literal["cpu", "cuda"] = "cpu"
    seed: int = 1337
    log_every_n_steps: int = 50


class WandbCfg(BaseModel):
    enable: bool = True
    project: str = "mlproj"
    entity: Optional[str] = None
    tags: list[str] = []
    notes: Optional[str] = None


class RunCfg(BaseModel):
    # high-level run metadata
    name: Optional[str] = None
    group: Optional[str] = None


class Config(BaseModel):
    data: DataCfg
    model: ModelCfg
    optim: OptimCfg
    trainer: TrainerCfg
    wandb: WandbCfg = WandbCfg()
    run: RunCfg = RunCfg()
