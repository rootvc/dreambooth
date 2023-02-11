from typing import Optional

import torch
from pydantic import BaseModel


class Model(BaseModel):
    name: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    revision: Optional[str] = None


class HyperParams(BaseModel):
    dtype: torch.dtype = torch.float16
    gradient_accumulation_steps: int = 2

    # Model
    model: Model = Model()
    prior_prompt: str = "a photo of a person"
    prior_samples: int = 100
    batch_size: int = 8

    # Optimizer
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    # Training
    loading_workers: int = 4
    train_epochs: int = 1
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0

    # Validation
    validate_every: int = 50  # epochs
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 4
    validation_steps: int = 25

    class Config:
        arbitrary_types_allowed = True
