import random
from functools import cached_property
from typing import Union

import torch
import wandb
from diffusers import (
    DiffusionPipeline,
    StableDiffusionDepth2ImgPipeline,
    StableDiffusionPipeline,
)
from PIL import Image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

from dreambooth.params import TEST_PROMPTS, Class, HyperParams
from dreambooth.registry import CompiledModelsRegistry
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.shared import (
    image_transforms,
    images,
    main_process_only,
    patch_allowed_pipeline_classes,
)

TClipModels = tuple[
    CLIPProcessor,
    CLIPModel,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
]


class Tester:
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

    @torch.no_grad()
    def clip_models(self) -> TClipModels:
        processor = CLIPProcessor.from_pretrained(
            self.params.test_model, local_files_only=True
        )
        clip_model, text_model, vis_model = [
            CompiledModelsRegistry.get(
                klass,
                self.params.test_model,
                compile=True,
            )
            .to(self.accelerator.device, non_blocking=True)
            .eval()
            .requires_grad_(False)
            for klass in (
                CLIPModel,
                CLIPTextModelWithProjection,
                CLIPVisionModelWithProjection,
            )
        ]
        return (processor, clip_model, text_model, vis_model)

    @cached_property
    def test_images(self):
        transforms = image_transforms(
            self.params.model.resolution, augment=False, to_pil=True
        )
        return [transforms(Image.open(p)) for p in images(self.instance_class.data)]

    @cached_property
    def test_prompts(self):
        return self.params.eval_prompts
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
        processor, _, text_model, _ = self.clip_models()
        text_embed_inputs = [
            processor.tokenizer(
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
        processor, _, _, vis_model = self.clip_models()
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

    def pipeline(self, pipe: StableDiffusionPipeline) -> DiffusionPipeline:
        with patch_allowed_pipeline_classes():
            return StableDiffusionDepth2ImgPipeline.from_pretrained(
                self.params.model.name,
                **pipe.components,
                strength=self.params.test_strength,
            ).to(self.accelerator.device, torch_dtype=torch.float)

    @main_process_only
    @torch.no_grad()
    def validate(self, pipe: StableDiffusionPipeline, title: str) -> list[Image.Image]:
        pipeline = self.pipeline(pipe)
        generator = torch.Generator(device=self.accelerator.device)
        images = pipeline(
            self.test_prompts,
            num_inference_steps=self.params.validation_steps,
            image=[self.test_images[0]] * len(self.test_prompts),
            guidance_scale=self.params.validation_guidance_scale,
            negative_prompt=[self.params.negative_prompt] * len(self.test_prompts),
            generator=generator,
        ).images
        images = list(images)
        self.log_images(
            self.test_prompts,
            images,
            title=f"{title}-{self.params.validation_guidance_scale}",
        )
        return images

    @main_process_only
    @torch.no_grad()
    def test_pipe(self, pipe: StableDiffusionPipeline, title: str):
        processor, _, _, vis_model = self.clip_models()
        images = self.validate(pipe, title)
        return {
            "text_alignments": 0.0,
            "text_alignment": 0.0,
            "image_alignments": 0.0,
            "image_alignment": 0.0,
        }

        image_embed_inputs = [
            processor(images=image, return_tensors="pt").to(self.accelerator.device)
            for image in images
        ]
        image_embeds = [vis_model(**inp).image_embeds for inp in image_embed_inputs]
        image_embeds = torch.cat(image_embeds, dim=0)

        return self.score(image_embeds, self.text_embeds, self.image_embeds)
