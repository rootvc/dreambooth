from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from compel import Compel

if TYPE_CHECKING:
    pass


@dataclass
class Compels:
    xl: Compel
    refiner: Compel


@dataclass
class Prompts:
    compels: Compels
    device: torch.device
    dtype: torch.dtype
    raw: list[str]
    negative: str

    def _to(self, d: dict):
        return {k: v.to(self.device, dtype=self.dtype) for k, v in d.items()}

    def _kwargs(self, compel: Compel, embeds: torch.Tensor, pools: torch.Tensor):
        neg_embeds, neg_pools = compel([self.negative] * len(self.raw))
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

    def kwargs_for_xl(self):
        embeds, pools = self.compels.xl(self.raw)
        return self._kwargs(self.compels.xl, embeds, pools)

    def kwargs_for_refiner(self):
        embeds, pools = self.compels.refiner(self.raw)
        return self._kwargs(self.compels.refiner, embeds, pools)
