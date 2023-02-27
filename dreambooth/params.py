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
    name: Union[str, Path] = "stabilityai/stable-diffusion-2-1"
    vae: Optional[Union[str, Path]] = "stabilityai/sd-vae-ft-mse"
    resolution: int = 768
    revision: Optional[str] = "fp16"


class HyperParams(BaseModel):
    dtype: torch.dtype = torch.float16
    gradient_accumulation_steps: int = 2

    # Model
    source_token: str = "person"
    token: str = "sks"
    model: Model = Model()
    prior_prompt: str = f"a photo of a {source_token}"
    prior_samples: int = 250
    prior_class: Optional[Class] = None
    batch_size: int = 1

    # Optimizer
    learning_rate: float = 1e-4
    text_learning_rate: float = 1e-4
    ti_learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    # Training
    dynamo_backend: Optional[str] = "inductor"
    loading_workers: int = 4
    ti_train_epochs: int = 15
    train_epochs: int = 100
    lr_scheduler: str = "linear"
    lr_warmup_steps: int = 300
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0

    # LoRA
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Text Encoder
    lora_text_rank: int = 8
    lora_text_alpha: int = 32
    lora_text_dropout: float = 0.1

    # Validation
    validate_after: int = 0  # 220  # steps
    validate_every: int = 10  # epochs
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 4
    validation_steps: int = 20
    negative_prompt: str = "poorly drawn hands, poorly drawn face, mutation, deformed, distorted, blurry, bad anatomy, bad proportions, extra limbs, cloned face"

    class Config:
        arbitrary_types_allowed = True
