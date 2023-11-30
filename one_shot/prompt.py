from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from compel import Compel

if TYPE_CHECKING:
    pass


@dataclass
class Compels:
    refiner: Compel
    inpainter: Compel
    base: Compel | None = None
    xl: Compel | None = None


@dataclass
class Prompts:
    compels: Compels
    device: torch.device
    dtype: torch.dtype
    raw: list[str]
    negative: list[str]

    def _to(self, d: dict):
        return {k: v.to(self.device, dtype=self.dtype) for k, v in d.items()}

    def _base_kwargs(self, compel: Compel, i: int):
        embeds = compel(self.raw[i])
        neg_embeds = compel(self.negative[i])
        [
            embeds,
            neg_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length([embeds, neg_embeds])
        return self._to({"prompt_embeds": embeds, "negative_prompt_embeds": neg_embeds})

    def _xl_kwargs(self, compel: Compel, i: int):
        embeds, pools = compel(self.raw[i])
        neg_embeds, neg_pools = compel(self.negative[i])
        [
            embeds,
            neg_embeds,
        ] = compel.pad_conditioning_tensors_to_same_length([embeds, neg_embeds])
        return self._to(
            {
                "prompt_embeds": embeds,
                "pooled_prompt_embeds": pools,
                "negative_prompt_embeds": neg_embeds,
                "negative_pooled_prompt_embeds": neg_pools,
            }
        )

    def kwargs_for_base(self, i: int):
        return self._base_kwargs(self.compels.base, i)

    def kwargs_for_xl(self, i: int):
        return self._xl_kwargs(self.compels.xl, i)

    def kwargs_for_refiner(self, i: int):
        return self._xl_kwargs(self.compels.refiner, i)

    def kwargs_for_inpainter(self, i: int):
        return self._xl_kwargs(self.compels.inpainter, i)
