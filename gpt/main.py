from typing import Optional, Any, Sequence, List
import os
import math
import yaml
import shutil

import torch
import coolname
import hydra
import pydantic
from omegaconf import DictConfig


class PretrainConfig(pydantic.BaseModel):
    # data
    data_path: str


@hydra.main(config_path="config", config_name="cfg_main", version_base=None)
def launch(hydra_config: DictConfig):
    pass


if __name__ == "__main__":
    launch()
