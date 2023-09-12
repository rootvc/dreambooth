from functools import cached_property
from typing import TYPE_CHECKING

import torch
from compel import Compel, ReturnedEmbeddingsType

if TYPE_CHECKING:
    from one_shot.ensemble import StableDiffusionXLAdapterEnsemblePipeline


class Prompts:
    def __init__(
        self,
        pipeline: "StableDiffusionXLAdapterEnsemblePipeline",
        raw: list[str],
        negative: str,
    ):
        self.pipe = pipeline
        self.raw = raw
        self.negative = negative

    @cached_property
    def xl_compel(self):
        return Compel(
            [self.pipe.tokenizer, self.pipe.tokenizer_2],
            [self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=self.pipe.device,
        )

    @cached_property
    def refiner_compel(self):
        return Compel(
            tokenizer=self.pipe.refiner.tokenizer_2,
            text_encoder=self.pipe.refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=self.pipe.refiner.device,
        )

    def _to(self, d: dict):
        return {
            k: v.to(self.pipe.device, dtype=self.pipe.text_encoder.dtype)
            for k, v in d.items()
        }

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
        embeds, pools = self.xl_compel(self.raw)
        return self._kwargs(self.xl_compel, embeds, pools)

    def kwargs_for_refiner(self):
        embeds, pools = self.refiner_compel(self.raw)
        return self._kwargs(self.refiner_compel, embeds, pools)
