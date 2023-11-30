from pydantic import BaseModel

from one_shot.params.base import Model as BaseModel_
from one_shot.params.base import Params as BaseParams
from one_shot.params.prompts import (
    F,
    P,
    PromptStrings,
    PromptTemplates,
)


class IPAdapterFiles(BaseModel):
    adapter: str
    image_encoder: str
    image_encoder_config: str


class IPAdapter(BaseModel):
    repo: str
    files: IPAdapterFiles
    vae: str
    base: str


class Model(BaseModel_):
    ip_adapter: IPAdapter = IPAdapter(
        repo="h94/IP-Adapter",
        files=IPAdapterFiles(
            adapter="models/ip-adapter-full-face_sd15.safetensors",
            image_encoder="models/image_encoder/model.safetensors",
            image_encoder_config="models/image_encoder/config.json",
        ),
        vae="stabilityai/sd-vae-ft-mse",
        base="SG161222/Realistic_Vision_V4.0_noVAE",
    )


class Params(BaseParams):
    model: Model = Model()
    prompt_templates: PromptTemplates = PromptTemplates(
        eyes=PromptStrings(
            positives=[
                "{color} eyes",
                P("perfecteyes", 2),
                P("detailed pupils"),
                "subtle eyes",
                "natural eyes",
                "realistic eyes",
                F("{ethnicity} {gender}", 0.1),
                F("{prompt}", 0.8),
            ],
            negatives=[P("cross eyed", 3), "eyes closed", "extra eyes"],
        ),
        merge=PromptStrings(
            positives=[
                "realistic",
                "{prompt}",
                "4k",
                "cohesive",
                "detailed",
                "high quality",
            ],
            negatives=["blurry", "fuzzy", "disjointed", "low quality", "lowres"],
        ),
        details=PromptStrings(
            positives=["4k photo", "{prompt}", "closeup portrait"],
            negatives=[
                "monochrome",
                "lowres",
                "bad anatomy",
                "worst quality",
                "low quality",
            ],
        ),
    )

    guidance_scale: float = 12.5
    refine_guidance_scale: float = 7.5
    refiner_strength: float = 0.75
    inpainting_strength: float = 0.40
    conditioning_strength: tuple[float, float] = (0.70, 0.72)
    conditioning_factor: float = 0.85
