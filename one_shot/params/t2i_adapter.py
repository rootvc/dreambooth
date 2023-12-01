from one_shot.params.base import Model as BaseModel_
from one_shot.params.base import Params as BaseParams
from one_shot.params.prompts import (
    F,
    M,
    P,
    PromptStrings,
    PromptTemplates,
)


class Model(BaseModel_):
    t2i_adapter: str = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
    detector: str = "lllyasviel/Annotators"


class Params(BaseParams):
    model: Model = Model()
    prompt_templates: PromptTemplates = PromptTemplates(
        background=PromptStrings(
            positives=[
                "closeup portrait",
                "{prompt}",
                "{ethnicity} {gender}",
                "4k",
                "realistic",
                "cinematic",
                "cinematic effect",
                P("hyperrealistic"),
                P("contrasts", 2),
                "sharp",
                P("highly detailed"),
            ],
            negatives=["cropped", "worst quality", "low quality"],
        ),
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
                "4k",
                "cohesive",
                "detailed",
                F("{prompt}", 0.2),
                F("{ethnicity} {gender}", 0.2),
            ],
            negatives=[P("split", 3), "blurry", "fuzzy", "disjointed", "eyes"],
        ),
        details=PromptStrings(
            positives=[
                "{prompt}",
                "4k",
                "beautiful",
                "cinematic",
                "cinematic effect",
                "good teeth",
                F("{ethnicity} {gender}", 0.5),
                P("hyperrealistic"),
                P("contrasts", 2),
                "sharp",
                P("highly detailed", 3),
                "white teeth",
                "nice smile",
                F("airbrushed", 0.2),
                F("beautiful", 0.2),
                F("dream", 0.2),
            ],
            negatives=[
                P("extra fingers", 2),
                P("ugly", 2),
                P("blurry", 3),
                P("incomplete", 2),
                P("fuzzy"),
                F("picture frame", 1.5),
                "ugly teeth",
                "disjointed",
                "monotone",
                "dreary",
                "extra digit",
                "eyes closed",
                "extra eyes",
                "bad smile",
                M("cropped"),
                M("worst quality"),
                M("low quality"),
                "glitch",
                "deformed",
                "mutated",
                "disfigured",
                "colored teeth",
                "yellow teeth",
            ],
        ),
    )

    guidance_scale: float = 9.0
    refine_guidance_scale: float = 9.0
    refiner_strength: float = 0.05
    inpainting_strength: float = 0.40
    conditioning_strength: tuple[float, float] = (1.5, 1.5)
    conditioning_factor: float = 1.0
    high_noise_frac: float = 0.85
