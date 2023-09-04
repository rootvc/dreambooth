import random
from pathlib import Path
from typing import Literal, Optional, Union

import torch
from pydantic import BaseModel

from dreambooth.param.model import Model

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
    IMAGENET_TEMPLATES: list[str] = [
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


class HyperParams(BaseModel):
    dtype: torch.dtype = torch.bfloat16
    gradient_accumulation_steps: int = 1

    # Model
    source_token: str = "person"
    token: str = "<krk>"
    model: Model = Model()
    prior_prompt: str = f"a closeup portrait photo of a single {source_token}, looking forward, without glasses, blank background"
    prior_samples: int = 250
    prior_class: Optional[Class] = None
    batch_size: int = 1

    # Optimizer
    learning_rate: float = 1e-5 / batch_size
    text_learning_rate: float = 4e-5 / batch_size
    ti_learning_rate: float = 0.0 / batch_size
    ti_continued_learning_rate: float = 0.0 / batch_size
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2

    # Training
    dynamo_backend: Optional[str] = None
    use_diffusers_unet: bool = False
    loading_workers: int = 4
    ti_train_epochs: int = 0
    lora_train_epochs: int = 10
    lr_scheduler: str = "cosine_with_restarts"
    lr_warmup_steps: int = 0
    lr_cycles: int = 3
    prior_loss_weight: float = 1.0
    max_grad_norm: float = 1.0
    snr_gamma: float = 5.0
    input_perterbation: float = 0.000

    # LoRA
    lora_rank: int = 32
    lora_alpha: float = 0.70
    lora_dropout: float = 0.1

    # Text Encoder
    lora_text_rank: int = 16
    lora_text_alpha: float = 1.0
    lora_text_dropout: float = 0.1

    # Validation
    validate_after_steps: int = 2500
    validate_every_epochs: Optional[dict] = {2500: 1}
    validation_prompt_suffix: str = "in a cowboy costume"
    validation_samples: int = 2
    validation_steps: int = 75
    validation_guidance_scale: float = 18.5
    # negative_prompt: str = "(((<bad_dream>), (<unreal_dream>)).and())+, (<all_negative>)+, (eyes closed, 'poorly drawn face, bad smile', 'chubby, fat, big head', 'deformed, disgusting, ugly, twisted').and()"
    negative_prompt: str = ", ".join(
        [
            "poorly drawn face",
            "elderly",
            "disgusting",
            "scary",
            "distorted",
            "disfigured",
            "deformed",
            "twisted",
            "grainy",
            "unfocused",
            "eyes closed",
            "bad smile",
            "ugly",
            "fat",
            "chubby",
        ]
    )
    test_model: Union[str, Path] = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    image_alignment_threshold: float = 0.82
    text_alignment_threshold: float = 0.21

    final_image_alignment_threshold: float = 0.60
    final_text_alignment_threshold: float = 0.20

    # Eval
    eval_prompts: list[str] = [
        f"a closeup portrait photo of a single {token} as a pirate",
        f"a closeup portrait photo of a single {token}, in a mech suit",
        f"8k portrait of {token}, wearing a colorful suit",
        f"an animation of {token}, a character from Naruto",
    ]
    # eval_prompts: list[str] = [
    #     f"{p}, (vibrant colors, masterpiece, sharp focus, detailed face).and(), (<add_details>)0.3, (<enhancer>)0.6"
    #     for p in [
    #         f"(('a closeup picture of ({token})+++', 'a closeup picture of a zombie').blend(0.7, 0.3), '(decaying skin and clothing)+++, (rotting)+, inside an (abandoned building)+').and()",
    #         f"(('a closeup portrait of ({token})+++', 'a closeup portrait of a Harry Potter character').blend(0.7, 0.3), (wearing robes and holding a wand)++, (in front of Hogwarts castle)++).and()",
    #         f"(close up Portrait photo of ({token})+++ in a clown costume, 'clown face makeup, red nose', (bright, colorful and vibrant)+).and()",
    #         f"(close up Portrait photo of ({token})+++ in a mech suit, 'light bokeh, intricate, steel metal, elegant, photo by greg rutkowski, soft lighting').and()",
    #         # f"('8k portrait of {token}, (wearing a colorful suit)++', (pop art style)+, clear and vibrant).and()",
    #         # f"(an (animation)+ of {token}, a character (from Naruto)+++, '(anime)++, colorful').and()",
    #         # f"('an (oil painting)+++ of {token}, by van gogh', (starry night sky in background)+, (vibrant)+).and()",
    #         # f"(a photograph of {token}+ the (Marvel superhero)++, '(cape and costume)+, flying in the sky, (nyc skyline in background)+', 'sharp and focused, realistic, strong').and()",
    #         # f"(a (cartoon)+++ screenshot of {token}, (clouds and sky in background)+++, 'wide angle shot, sharp').and()",
    #         # f"('(3d render)++ of {token}+, as an (cyborg)++', (robot body parts)+, (cyberpunk)++ lighting).and()",
    #     ]
    # ]

    debug_outputs: bool = True
    test_steps: int = 30
    test_images: int = 4
    test_guidance_scale: float = 7.5
    test_strength: float = 0.75
    high_noise_frac: float = 0.8
    mask_padding: float = 0.15
    eval_model_path: Path = Path("CodeFormer")
    model_output_path: Path = Path("output/model")
    image_output_path: Path = Path("output/images")

    class Config:
        arbitrary_types_allowed = True
