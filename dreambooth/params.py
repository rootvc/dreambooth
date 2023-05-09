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
    name: Union[str, Path] = "stabilityai/stable-diffusion-2-depth"
    vae: Optional[Union[str, Path]] = "stabilityai/sd-vae-ft-mse"
    resolution: int = 512
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
    learning_rate: float = 1e-3
    text_learning_rate: float = 2e-3
    ti_learning_rate: float = 5e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    epsilon: float = 1e-8

    # Training
    dynamo_backend: Optional[str] = None
    use_diffusers_unet: bool = False
    loading_workers: int = 4
    train_epochs: int = 6
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 0
    lr_cycles: int = 9
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    snr_gamma: float = 5.0
    input_perterbation: float = 0.10

    # LoRA
    lora_rank: int = 16
    lora_alpha = 0.10
    lora_dropout: float = 0.1

    # Text Encoder
    lora_text_rank: int = 80
    lora_text_alpha: float = 1.0
    lora_text_dropout: float = 0.1

    # Validation
    validate_after_steps: int = 2500
    validate_every_epochs: Optional[dict] = {2500: 1}
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 2
    validation_steps: int = 75
    validation_guidance_scale: float = 18.5
    negative_prompt: str = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft, eyes closed"
    test_model: Union[str, Path] = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    image_alignment_threshold: float = 0.82
    text_alignment_threshold: float = 0.21

    final_image_alignment_threshold: float = 0.60
    final_text_alignment_threshold: float = 0.20

    # Eval
    eval_prompts: list[str] = [
        f"a closeup portrait of {token}, as a zombie, decaying skin and clothing, dark and eerie, highly detailed, photorealistic, 8k, ultra realistic, horror style, art by greg rutkowski, charlie bowater, and magali villeneuve.",
        f"a closeup portrait of {token}, as a Harry Potter character, magical world, wands, robes, Hogwarts castle in the background, enchanted forest, detailed lighting, art by jim kay, charlie bowater, alphonse mucha, ronald brenzell, digital painting, concept art.",
        f"Closeup portrait of {token}, as a clown, highly detailed, surreal, expressionless face, bright colors, contrast lighting, abstract background, art by wlop, greg rutkowski, charlie bowater, magali villeneuve, alphonse mucha, cartoonish, comic book style.",
        f"8k portrait of {token}, pop art style, incredibly detailed faces, wearing a colorful men's suit, 🎨🖌️, idol, ios",
        f"closeup portrait of {token} as a superhero, dynamic lighting, intense colors, detailed costume, artstation trending, art by alphonse mucha, greg rutkowski, ross tran, leesha hannigan, ignacio fernandez rios, kai carpenter, noir photorealism, film",
    ]

    upscale_model = "stabilityai/sd-x2-latent-upscaler"
    upscale_factor: int = 1
    restore_faces: bool = False
    debug_outputs: bool = False
    fidelity_weight: float = 0.5
    test_steps: int = 120
    test_guidance_scale: float = 16.5
    test_strength = 0.70  # 0.70
    eval_model_path: Path = Path("CodeFormer")
    model_output_path: Path = Path("output/model")
    image_output_path: Path = Path("output/images")

    class Config:
        arbitrary_types_allowed = True
