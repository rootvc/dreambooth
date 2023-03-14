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
    "{} oil painting ghibli inspired",
    "{} working on the laptop",
    "{} with mountains and sunset in background",
    "Painting of {} at a beach by artist claude monet",
    "{} digital painting 3d render geometric style",
    "A screaming {}",
    "A depressed {}",
    "A sleeping {}",
    "A sad {}",
    "A joyous {}",
    "A frowning {}",
    "A sculpture of {}",
    "{} near a pool",
    "{} at a beach with a view of seashore",
    "{} in a garden",
    "{} in grand canyon",
    "{} floating in ocean",
    "{} and an armchair",
    "A maple tree on the side of {}",
    "{} and an orange sofa",
    "{} with chocolate cake on it",
    "{} with a vase of rose flowers on it",
    "A digital illustration of {}",
    "Georgia O'Keeffe style {} painting",
    "A watercolor painting of {} on a beach",
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
    name: Union[str, Path] = "stabilityai/stable-diffusion-2-1"
    vae: Optional[Union[str, Path]] = "stabilityai/sd-vae-ft-mse"
    resolution: int = 768
    revision: Optional[str] = "fp16"


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
    learning_rate: float = 1e-5
    text_learning_rate: float = 1e-5
    ti_learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    # Training
    dynamo_backend: Optional[str] = None
    use_diffusers_unet: bool = False
    loading_workers: int = 4
    ti_train_epochs: int = 15
    train_epochs: int = 60
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 150
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
    validate_after_steps: int = 400
    validate_every_epochs: Optional[dict] = {400: 3, 500: 2, 600: 1}
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 3
    validation_steps: int = 50
    validation_guidance_scale: float = 8.5
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"
    test_model: Union[str, Path] = "openai/clip-vit-large-patch14"

    image_alignment_threshold: float = 0.75
    text_alignment_threshold: float = 0.20

    # Eval
    eval_prompts: list[str] = [
        f"a photo of a {token}, in a cowboy costume",
        f"a photo of a {token}, as a viking, ultra realistic, concept art, intricate details, powerful and fierce, highly detailed, octane render, 8 k, art by artgerm and greg rutkowski and charlie bowater and magali villeneuve and alphonse mucha, golden hour, horns and braids in hair, fur-lined cape and helmet, axe in hand",
        f"a photo of a {token}, as a viking, ultra realistic, intricate details, powerful and fierce, art by artgerm and greg rutkowski and charlie bowater and magali villeneuve and alphonse mucha",
        f"a photo of a {token}, as a viking, golden hour, horns and braids in hair, fur-lined cape and helmet, with axe in hand",
        f"closeup portrait painting of a {token}, as a viking, ultra realistic, concept art, intricate details, powerful and fierce, highly detailed, octane render, 8 k, art by artgerm and greg rutkowski and charlie bowater and magali villeneuve and alphonse mucha, golden hour, horns and braids in hair, fur-lined cape and helmet, axe in hand",
        f"closeup portrait of a {token}, as a paladin, wearing brilliant white armor and a crown, fantasy concept art, artstation trending, highly detailed, beautiful landscape in the background, art by wlop, greg rutkowski, thierry doizon, charlie bowater, alphonse mucha, golden hour lighting, ultra realistic.",
        f"closeup portrait of a {token}, as a Harry Potter character, magical world, wands, robes, Hogwarts castle in the background, enchanted forest, detailed lighting, art by jim kay, charlie bowater, alphonse mucha, ronald brenzell, digital painting, concept art.",
        f"closeup portrait of a {token}, as a clown, highly detailed, surreal, expressionless face, bright colors, contrast lighting, abstract background, art by wlop, greg rutkowski, charlie bowater, magali villeneuve, alphonse mucha, cartoonish, comic book style.",
        # f"closeup portrait of a {token}, as a jedi with a lightsaber, highly detailed, science fiction, star wars concept art, intricate details, bright colors, golden hour, art by marko djurdjevic, greg rutkowski, wlop, fredperry, digital painting, rossdraws.",
        # f"closeup portrait of a {token}, as a ninja, wearing a black hood and suit, stealthy movements, dark night background, shadows and mist, detailed and realistic, art by kazuya yamashita, yuya kanzaki, yang zhizhuo, digital painting, photorealism, 8k resolution.",
    ]

    upscale_factor: int = 2
    fidelity_weight: float = 0.5
    test_steps: int = 150
    test_guidance_scale: float = 8.5
    eval_model_path: Path = Path("CodeFormer")
    model_output_path: Path = Path("output/model")
    image_output_path: Path = Path("output/images")

    class Config:
        arbitrary_types_allowed = True
