import random
from functools import cached_property
from typing import ClassVar, Union

import torch
import wandb
from diffusers import (
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from dreambooth.params import TEST_PROMPTS, Class, HyperParams
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.shared import compile_model, main_process_only

TClipModels = tuple[
    CLIPTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
]


class Tester:
    _clip_models: ClassVar[TClipModels] = None

    def __init__(
        self, params: HyperParams, accelerator: BaseAccelerator, instance_class: Class
    ):
        self.params = params
        self.accelerator = accelerator
        self.instance_class = instance_class

    @main_process_only
    def log_images(
        self,
        prompts: Union[str, list[str]],
        images: Union[list[Image.Image], list[str]],
        title: str = "validation",
    ):
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        self.accelerator.wandb_tracker.log(
            {
                title: [
                    wandb.Image(image, caption=f"{i}: {prompts[i]}")
                    for i, image in enumerate(images)
                ]
            }
        )

    @staticmethod
    @torch.inference_mode()
    def prepare_clip_model_sets(model) -> TClipModels:
        text_model = CLIPTextModelWithProjection.from_pretrained(model)
        tokenizer = CLIPTokenizer.from_pretrained(model)
        vis_model = CLIPVisionModelWithProjection.from_pretrained(model)
        processor = CLIPProcessor.from_pretrained(model)

        return tuple(
            [tokenizer, processor]
            + [
                compile_model(
                    m.to("cuda").eval().requires_grad_(False),
                )
                for m in (text_model, vis_model)
            ]
        )

    def clip_models(self):
        if not self.__class__._clip_models:
            self.__class__._clip_models = self.prepare_clip_model_sets(
                self.params.test_model
            )
        return self.__class__._clip_models

    @cached_property
    def test_images(self):
        return [Image.open(p) for p in self.instance_class.data.iterdir()]

    @cached_property
    def test_prompts(self):
        prompt = (
            self.instance_class.deterministic_prompt
            + ", "
            + self.params.validation_prompt_suffix
        )
        return [prompt] + [
            p.format(self.params.token)
            for p in random.sample(TEST_PROMPTS, self.params.validation_samples - 1)
        ]

    @cached_property
    def text_embeds(self):
        tokenizer, _, text_model, _ = self.clip_models()
        text_embed_inputs = [
            tokenizer(
                p.replace(self.params.token, self.params.source_token),
                padding=True,
                return_tensors="pt",
            ).to(self.accelerator.device)
            for p in self.test_prompts
        ]
        text_embeds = [text_model(**inp).text_embeds for inp in text_embed_inputs]
        return torch.cat(text_embeds, dim=0)

    @cached_property
    def image_embeds(self):
        _, processor, _, vis_model = self.clip_models()
        target_image_embed_inputs = processor(
            images=self.test_images, return_tensors="pt"
        ).to(self.accelerator.device)
        return vis_model(**target_image_embed_inputs).image_embeds

    @main_process_only
    def score(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        target_image_embeds: torch.Tensor,
    ):
        assert image_embeds.shape[0] == text_embeds.shape[0]
        text_image_similarity = (image_embeds * text_embeds).sum(dim=-1) / (
            image_embeds.norm(dim=-1) * text_embeds.norm(dim=-1)
        )

        image_embeds_normalized = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        average_target_image_embed = (
            (target_image_embeds / target_image_embeds.norm(dim=-1, keepdim=True))
            .mean(dim=0)
            .unsqueeze(0)
            .repeat(image_embeds.shape[0], 1)
        )

        image_image_similarity = (
            image_embeds_normalized * average_target_image_embed
        ).sum(dim=-1)

        return {
            "text_alignments": wandb.Histogram(text_image_similarity.detach().tolist()),
            "text_alignment": text_image_similarity.mean().detach().item(),
            "image_alignments": wandb.Histogram(
                image_image_similarity.detach().tolist()
            ),
            "image_alignment": image_image_similarity.mean().detach().item(),
        }

    @main_process_only
    def validate(self, pipe: StableDiffusionPipeline, title: str) -> list[Image.Image]:
        generator = torch.Generator(device=self.accelerator.device)

        images = pipe(
            self.test_prompts,
            num_inference_steps=self.params.validation_steps,
            guidance_scale=self.params.validation_guidance_scale,
            negative_prompt=[self.params.negative_prompt] * len(self.test_prompts),
            generator=generator,
        ).images

        images = list(images)
        self.log_images(self.test_prompts, images, title=title)
        return images

    @main_process_only
    @torch.no_grad()
    def test_pipe(self, pipe: StableDiffusionPipeline, title: str):
        _, processor, _, vis_model = self.clip_models()
        images = self.validate(pipe, title)

        image_embed_inputs = [
            processor(images=image, return_tensors="pt").to(self.accelerator.device)
            for image in images
        ]
        image_embeds = [vis_model(**inp).image_embeds for inp in image_embed_inputs]
        image_embeds = torch.cat(image_embeds, dim=0)

        return self.score(image_embeds, self.text_embeds, self.image_embeds)
