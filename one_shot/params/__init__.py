from .base import Params as _Params
from .ip_adapter import Params as IPAdapterParams  # noqa
from .prompts import PromptStrings, PromptTemplates
from .settings import Settings as Settings
from .t2i_adapter import Params as T2IAdapterParams  # noqa


class Params(_Params):
    prompt_templates: PromptTemplates = PromptTemplates(
        eyes=PromptStrings(positives=[], negatives=[]),
        merge=PromptStrings(positives=[], negatives=[]),
        details=PromptStrings(positives=[], negatives=[]),
    )
    guidance_scale: float = 0.0
    refine_guidance_scale: float = 0.0
    refiner_strength: float = 0.0
    inpainting_strength: float = 0.0
    conditioning_strength: tuple[float, float] = (0.0, 0.0)
    conditioning_factor: float = 0.0
