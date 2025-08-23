# mlproj/train.py
from typing import Dict, Any, Optional
import random, numpy as np, time

try:
    import torch
except ImportError:
    torch = None

from omegaconf import DictConfig
from .schemas import Config

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def make_dataloader(data_cfg):
    # Pseudocode; replace with real datasets
    size = 50000 if data_cfg.name == "cifar10" else 60000
    batch_size = data_cfg.batch_size
    steps = size // batch_size
    return steps

def make_model(model_cfg, device: str):
    # Pseudocode; replace with real torch.nn.Module
    class DummyModel:
        def __init__(self, dim):
            self.dim = dim
        def train_step(self):
            # fake decreasing loss
            return max(0.1, 2.0 / (self.dim + 1))
    dim = model_cfg.hidden_dim
    return DummyModel(dim)

def train(cfg: DictConfig, pcfg: Config, wandb_run=None) -> Dict[str, float]:
    set_seed(pcfg.trainer.seed)
    device = pcfg.trainer.device

    steps_per_epoch = make_dataloader(pcfg.data)
    model = make_model(pcfg.model, device)

    global_step = 0
    loss = 2.0
    best = float("inf")

    for epoch in range(pcfg.trainer.max_epochs):
        for _ in range(steps_per_epoch):
            global_step += 1
            # pretend the loss decreases slowly
            loss = loss * 0.995 + model.train_step() * 0.005

            if wandb_run and global_step % pcfg.trainer.log_every_n_steps == 0:
                wandb_run.log({
                    "train/loss": loss,
                    "epoch": epoch + (global_step / steps_per_epoch),
                    "global_step": global_step,
                })

        best = min(best, loss)
        print(f"Epoch {epoch+1}/{pcfg.trainer.max_epochs} - loss={loss:.4f}")

    return {"loss": loss, "best_loss": best}
