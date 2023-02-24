from pathlib import Path
from typing import Optional, Union

import torch
from pydantic import BaseModel


class Class(BaseModel):
    prompt: str
    data: Path

    def check(self):
        return self.data.exists()


class Model(BaseModel):
    name: Union[str, Path] = "stabilityai/stable-diffusion-2-1-base"
    vae: Optional[Union[str, Path]] = "stabilityai/sd-vae-ft-mse"
    resolution: int = 512
    revision: Optional[str] = "fp16"


class HyperParams(BaseModel):
    dtype: torch.dtype = torch.float16
    gradient_accumulation_steps: int = 2

    # Model
    model: Model = Model()
    prior_prompt: str = "a photo of a person"
    prior_samples: int = 250
    prior_class: Optional[Class] = None
    batch_size: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    # Training
    loading_workers: int = 4
    train_epochs: int = 50
    lr_scheduler: str = "linear"
    lr_warmup_steps: int = 500
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Text Encoder
    train_text_encoder: bool = True
    lora_text_rank: int = 8
    lora_text_alpha: int = 32
    lora_text_dropout: float = 0.1

    # Validation
    validate_every: int = 5  # epochs
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 4
    validation_steps: int = 25
    negative_prompt: str = "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face"

    class Config:
        arbitrary_types_allowed = True
