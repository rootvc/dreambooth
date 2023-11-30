import random
from dataclasses import dataclass, field
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import numpy as np
import torch
import torch.multiprocessing
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from ip_adapter import IPAdapterFull as _IPAdapterFull
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from PIL import Image
from torchvision.transforms.functional import (
    to_pil_image,
)

from one_shot.face import Bounds, FaceHelper, FaceHelperModels
from one_shot.params import Params, PromptStrings, Settings
from one_shot.prompt import Compels, Prompts
from one_shot.utils import (
    Face,
    civitai_path,
    collect,
    dilate_mask,
    erode_mask,
    exclude,
    load_hf_file,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process import ProcessRequest
    from one_shot.logging import PrettyLogger


class IPAdapterFull(_IPAdapterFull):
    def generate(
        self,
        image: Image.Image,
        scale: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
        num_inference_steps: int = 30,
        **kwargs,
    ):
        image = image.resize((224, 224))

        self.set_scale(scale)
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(image)
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(
            1, num_samples, 1
        )
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(
            bs_embed * num_samples, seq_len, -1
        )

        prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
        )

        return self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images


@dataclass
class SharedModels:
    face: FaceHelperModels


@dataclass
class ProcessModels:
    base: StableDiffusionXLPipeline
    pipe: StableDiffusionXLCustomPipeline
    inpainter: StableDiffusionXLInpaintPipeline
    refiner: StableDiffusionXLInpaintPipeline
    ip_adapter: IPAdapterFull
    compels: Compels
    face: FaceHelperModels
    settings: Settings = Settings()

    @classmethod
    def _load_loras(cls, params: Params, pipe, key: str = "base"):
        for repo, lora in params.model.loras[key].items():
            if lora == "civitai":
                path = civitai_path(repo)
                pipe.load_lora_weights(
                    str(path.parent),
                    weight_name=str(path.name),
                    **cls.settings.loading_kwargs,
                )
            else:
                pipe.load_lora_weights(
                    repo, weight_name=lora, **cls.settings.loading_kwargs
                )
        pipe.fuse_lora(lora_scale=params.lora_scale)

    @classmethod
    @torch.inference_mode()
    def load(cls, params: Params, rank: int):
        base = StableDiffusionXLPipeline.from_pretrained(
            params.model.name,
            vae=AutoencoderKL.from_pretrained(
                params.model.vae,
                **exclude(cls.settings.loading_kwargs, {"variant"}),
            ).to(rank),
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(
                params.model.name, subfolder="scheduler"
            ),
            **cls.settings.loading_kwargs,
        ).to(rank)
        base.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, base)

        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        vae = AutoencoderKL.from_pretrained(
            params.model.ip_adapter.vae,
            **exclude(cls.settings.loading_kwargs, {"variant"}),
        ).to(rank)
        pipe = StableDiffusionPipeline.from_pretrained(
            params.model.ip_adapter.base,
            scheduler=noise_scheduler,
            vae=vae,
            feature_extractor=None,
            safety_checker=None,
            **exclude(cls.settings.loading_kwargs, {"variant"}),
        ).to(rank)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.enable_xformers_memory_efficient_attention()

        ip_adapter = IPAdapterFull(
            pipe,
            Path(
                load_hf_file(
                    params.model.ip_adapter.repo,
                    params.model.ip_adapter.files.image_encoder,
                )
            ).parent,
            load_hf_file(
                params.model.ip_adapter.repo, params.model.ip_adapter.files.adapter
            ),
            rank,
            num_tokens=257,
        )

        inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.inpainter,
            text_encoder=base.text_encoder,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank, params.dtype)

        # inpainter.unet = torch.compile(
        #     inpainter.unet, mode="reduce-overhead", fullgraph=True
        # )
        inpainter.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, inpainter, key="inpainter")

        refiner = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.refiner,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        # refiner.unet = torch.compile(
        #     refiner.unet, mode="reduce-overhead", fullgraph=True
        # )
        refiner.enable_xformers_memory_efficient_attention()

        base_compel = Compel(
            pipe.tokenizer,
            pipe.text_encoder,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
            requires_pooled=False,
            device=rank,
        )
        xl_compel = Compel(
            [base.tokenizer, base.tokenizer_2],
            [base.text_encoder, base.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
        )
        inpainter_compel = Compel(
            [inpainter.tokenizer, inpainter.tokenizer_2],
            [inpainter.text_encoder, inpainter.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
            device=rank,
        )
        refiner_compel = Compel(
            tokenizer=refiner.tokenizer_2,
            text_encoder=refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=rank,
        )
        return cls(
            base=base,
            pipe=pipe,
            ip_adapter=ip_adapter,
            refiner=refiner,
            inpainter=inpainter,
            compels=Compels(
                base=base_compel,
                xl=xl_compel,
                refiner=refiner_compel,
                inpainter=inpainter_compel,
            ),
            face=FaceHelperModels.load(params, rank),
        )


@dataclass
class Model:
    params: Params
    rank: int
    models: ProcessModels
    logger: "PrettyLogger"
    settings: Settings = Settings()

    @torch.inference_mode()
    def run(self, request: "ProcessRequest") -> list[Image.Image]:
        return ModelInstance(self, request).generate()

    @torch.inference_mode()
    def tune(self, request: "ProcessRequest") -> list[Image.Image]:
        instance = ModelInstance(self, request)
        faces = instance.faces
        redo_background = instance._redo_background(faces)
        final_refine = instance._final_refine(redo_background)
        return [
            instance.request.generation.images[0],
            # instance.faces[0],
            # instance.tmp[0],  # face mask
            # instance.tmp[1],  # face mask
            # instance.faces[0],
            # instance.tmp[2].convert("RGB"),  # masked face
            # instance.controls[0].convert("RGB"),
            # instance.backgrounds[0],
            # ,
            # instance.outpaint_bases[0],
            # instance.masks[0].convert("RGB"),
            # instance.og_masks[0].convert("RGB"),
            # instance.edge_masks[0].convert("RGB"),
            # #
            faces[0],
            # smooth_edges[0],
            redo_background[0],
            final_refine[0],
        ]


@dataclass
class ModelInstance:
    model: Model
    request: "ProcessRequest"
    tmp: list[Image.Image] = field(default_factory=list)

    def face_helper(self, images: list[Image.Image], tag: str = "", **kwargs):
        return FaceHelper(
            self.params,
            self.models.face,
            images,
            logger=self.model.logger.bind(tag=tag),
            **kwargs,
        )

    @property
    def rank(self):
        return self.model.rank

    @property
    def models(self):
        return self.model.models

    @property
    def params(self):
        return self.model.params

    def _prompts(self, strings: PromptStrings, **kwargs) -> Prompts:
        return Prompts(
            self.models.compels,
            self.rank,
            self.params.dtype,
            [
                strings.positive(prompt=p, **self.request.demographics, **kwargs)
                for p in self.request.generation.prompts
            ],
            [
                strings.negative(prompt=p, **self.request.demographics, **kwargs)
                for p in self.request.generation.prompts
            ],
        )

    @cached_property
    def background_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.background)

    def eyes_prompts(self, **kwargs) -> Prompts:
        return self._prompts(self.params.prompt_templates.eyes, **kwargs)

    @cached_property
    def merge_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.merge)

    @cached_property
    def details_prompts(self) -> Prompts:
        return self._prompts(self.params.prompt_templates.details)

    @cached_property
    def generator(self) -> torch.Generator | None:
        if self.model.params.seed:
            return torch.Generator(device=self.rank).manual_seed(self.params.seed)
        else:
            return None

    def _generate_face_images(self) -> list[Image.Image]:
        return [
            self.model.models.ip_adapter.generate(
                image=img,
                generator=self.generator,
                num_inference_steps=self.params.steps,
                guidance_scale=self.params.guidance_scale,
                control_guidance_start=1.0 - self.params.conditioning_factor,
                scale=random.triangular(*self.params.conditioning_strength),
                **self.details_prompts.kwargs_for_base(idx),
            )[0].resize(self.dims)
            for idx, img in enumerate(self.request.generation.images)
        ]

    def _touch_up_eyes(self, images: list[Image.Image]) -> list[Image.Image]:
        face_helper = self.face_helper(images, "eyes")
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.model.logger.info("Colors: {}", colors)
        return [
            self.models.inpainter(
                image=img,
                mask_image=masks[idx],
                generator=self.generator,
                strength=self.params.inpainting_strength,
                num_inference_steps=self.params.inpainting_steps,
                **self.eyes_prompts(color=colors[idx]).kwargs_for_inpainter(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    @cached_property
    def faces(self) -> list[Image.Image]:
        return self._touch_up_eyes(self._generate_face_images())

    @property
    def dims(self) -> tuple[int, int]:
        return (self.params.model.resolution, self.params.model.resolution)

    def _get_masks(
        self, frame: np.ndarray, face: Face
    ) -> tuple[Image.Image, Image.Image]:
        if face.mask:
            self.model.logger.warning("Using face mask")
            mask = ~face.mask.arr
            og_mask = ~erode_mask(face.mask.arr)
        elif not face.is_trivial:
            self.model.logger.warning("Using bounds mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[
                Bounds.from_face(frame.shape[:2], face).slice(self.params.mask_padding)
            ] = 0
            og_mask = np.full(frame.shape, 255, dtype=np.uint8)
            og_mask[Bounds.from_face(frame.shape[:2], face).slice()] = 0
            og_mask = ~erode_mask(~og_mask)
        else:
            self.model.logger.warning("Using default mask")
            mask = np.full(frame.shape, 255, dtype=np.uint8)
            mask[frame != 0] = 0
            og_mask = ~erode_mask(~mask)

        return tuple(to_pil_image(m, mode="RGB").convert("L") for m in (mask, og_mask))

    @cached_property
    @collect
    def _masks(self) -> Generator[tuple[Image.Image, Image.Image], None, None]:
        face_helper = self.face_helper(self.faces, "faces", conservative=False)
        bounds = face_helper.primary_face_bounds()
        for idx, face in enumerate(self.faces):
            yield self._get_masks(np.asarray(face), bounds[idx][1])

    @property
    def masks(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._masks))

    @property
    def og_masks(self) -> list[Image.Image]:
        return list(map(itemgetter(1), self._masks))

    @cached_property
    def edge_masks(self) -> list[Image.Image]:
        return [
            to_pil_image(
                (
                    dilate_mask(~np.asarray(self.masks[idx]))
                    - ~np.asarray(self.og_masks[idx])
                )
            )
            for idx in range(len(self._masks))
        ]

    def _redo_background(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.inpainter(
                image=image,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.refine_guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(images)
        ]

    def _final_refine(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.refiner(
                image=img,
                mask_image=self.masks[idx],
                generator=self.generator,
                strength=self.params.refiner_strength,
                guidance_scale=self.params.refine_guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    def generate(self) -> list[Image.Image]:
        redo_background = self._redo_background(self.faces)
        return self._final_refine(redo_background)
