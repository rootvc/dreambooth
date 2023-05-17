import random
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel

TEST_PROMPTS = [
    "{} swimming in a pool",
    "{} at a beach with a view of seashore",
    "{} in times square",
    "{} wearing sunglasses",
    "{} in a construction outfit",
    "{} playing with a ball",
    "{} wearing headphones",
    "an oil paining of {}, ghibli inspired",
    "{} working on the laptop",
    "{} with mountains and sunset in background",
    "A screaming {}",
    "A depressed {}",
    "A sleeping {}",
    "A sad {}",
    "A joyous {}",
    "A frowning {}",
    "A sculpture of {}",
    "a photo of {} near a pool",
    "a photo of {} at a beach with a view of seashore",
    "a photo of {} in a garden",
    "a photo of {} in grand canyon",
    "a photo of {} floating in ocean",
    "a photo of {} and an armchair",
    "{} and an orange sofa",
    "a photo of {} holding a vase of roses",
    "A digital illustration of {}",
    "Georgia O'Keeffe style painting of {}",
]


class Class(BaseModel):
    IMAGENET_TEMPLATES = [
        "a photo of a {}",
        "a rendering of a {}",
        "a cropped photo of the {}",
        "the photo of a {}",
        "a photo of a clean {}",
        "a photo of a dirty {}",
        "a dark photo of the {}",
        "a photo of my {}",
        "a photo of the cool {}",
        "a close-up photo of a {}",
        "a bright photo of the {}",
        "a cropped photo of a {}",
        "a photo of the {}",
        "a good photo of the {}",
        "a photo of one {}",
        "a close-up photo of the {}",
        "a rendition of the {}",
        "a photo of the clean {}",
        "a rendition of a {}",
        "a photo of a nice {}",
        "a good photo of a {}",
        "a photo of the nice {}",
        "a photo of the small {}",
        "a photo of the weird {}",
        "a photo of the large {}",
        "a photo of a cool {}",
        "a photo of a small {}",
    ]
    SUFFIXES: list[str] = ["", "4k", "highres", "high quality", "realistic"]

    prompt_: str
    type_: Literal["prompt", "token"] = "prompt"
    data: Path

    @property
    def prompt(self):
        if self.type_ == "prompt":
            return self.prompt_
        prompt = random.choice(self.IMAGENET_TEMPLATES).format(self.prompt_)
        suffix = random.choice(self.SUFFIXES)
        return ", ".join(filter(None, (prompt, suffix)))

    @property
    def deterministic_prompt(self):
        if self.type_ == "prompt":
            return self.prompt_
        return self.IMAGENET_TEMPLATES[0].format(self.prompt_)

    def check(self):
        return self.data.exists()


class Model(BaseModel):
    name: Union[str, Path] = "Lykon/DreamShaper"
    vae: Optional[Union[str, Path]] = "stabilityai/sd-vae-ft-mse"
    control_net: Optional[Union[str, Path]] = "lllyasviel/control_v11p_sd15_canny"
    resolution: int = 512
    revision: Optional[str] = None


class HyperParams(BaseModel):
    dtype: torch.dtype = torch.float16
    gradient_accumulation_steps: int = 1

    # Model
    source_token: str = "person"
    token: str = "<krk>"
    model: Model = Model()
    prior_prompt: str = f"a photo of a {source_token}"
    prior_samples: int = 250
    prior_class: Optional[Class] = None
    batch_size: int = 1

    # Optimizer
    learning_rate: float = 2e-3
    text_learning_rate: float = 6e-4
    ti_learning_rate: float = 1e-3
    ti_continued_learning_rate: float = 8e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2

    # Training
    dynamo_backend: Optional[str] = None
    use_diffusers_unet: bool = False
    loading_workers: int = 4
    ti_train_epochs: int = 4
    lora_train_epochs: int = 2
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 0
    lr_cycles: int = 3
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    snr_gamma: float = 5.0
    input_perterbation: float = 0.001

    # LoRA
    lora_rank: int = 24
    lora_alpha = 16.0
    lora_dropout: float = 0.01

    # Text Encoder
    lora_text_rank: int = 24
    lora_text_alpha: float = 10.0
    lora_text_dropout: float = 0.01

    # Validation
    validate_after_steps: int = 2500
    validate_every_epochs: Optional[dict] = {2500: 1}
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 2
    validation_steps: int = 75
    validation_guidance_scale: float = 18.5
    negative_prompt: str = "(ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, eyes closed)0.2"
    test_model: Union[str, Path] = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    image_alignment_threshold: float = 0.82
    text_alignment_threshold: float = 0.21

    final_image_alignment_threshold: float = 0.60
    final_text_alignment_threshold: float = 0.20

    # Eval
    eval_prompts: list[str] = [
        f"('a closeup portrait of {token}', 'a closeup portrait of a zombie').blend(0.10, 0.90), (decaying skin and clothing)++, (dark and imposing)---, (high contrast)+",
        #
        f"('a closeup portrait of {token}', 'a closeup portrait of a Harry Potter character').blend(0.05, 0.95), magical world, wands, robes, (Hogwarts castle in the background)+, (high contrast)+",
        #
        f"a closeup portrait of {token}, (as a clown)1.25, expressionless face, (bright, colorful and vibrant)++",
        #
        f"8k portrait of {token}, (pop art style)1.3, (wearing a colorful suit)+++, clear and vibrant",
        #
        f"a drawing of {token} (from Naruto)1.4, (anime style)+, colorful",
        #
        f"an (oil painting)+ of {token} by (van gogh)1.2",
        #
        f"('{token}', 'a beautiful Disney princess').blend(0.03, 0.97)",
        f"{token} as (a Disney princess)++++",
        #
        f"a cartoon portrait of {token}, (clouds and sky in background)1.3, (wide angle)-",
        #
        f"complex 3d render of {token}'s face, (robot face)1.3, (bright cyberpunk lighting)+++, 150 mm",
    ]

    debug_outputs: bool = True
    test_steps: int = 50
    test_guidance_scale: float = 20.0
    test_strength = 1.0  # 0.70
    mask_padding = 0.15
    eval_model_path: Path = Path("CodeFormer")
    model_output_path: Path = Path("output/model")
    image_output_path: Path = Path("output/images")

    class Config:
        arbitrary_types_allowed = True
