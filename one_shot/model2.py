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
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)
from ip_adapter import IPAdapterPlusXL as _IPAdapterPlusXL
from ip_adapter.custom_pipelines import StableDiffusionXLCustomPipeline
from PIL import Image, ImageOps
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
    draw_masks,
    erode_mask,
    exclude,
    load_hf_file,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.process import ProcessRequest
    from one_shot.logging import PrettyLogger


class IPAdapterPlusXL(_IPAdapterPlusXL):
    def generate(
        self,
        image: Image.Image,
        scale: float,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
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
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
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
    base_inpainter: StableDiffusionXLInpaintPipeline
    inpainter: StableDiffusionXLControlNetInpaintPipeline
    refiner: StableDiffusionXLInpaintPipeline
    ip_adapter: IPAdapterPlusXL
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

        pipe = StableDiffusionXLCustomPipeline.from_pretrained(
            params.model.name,
            add_watermarker=False,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        pipe.enable_xformers_memory_efficient_attention()
        cls._load_loras(params, pipe)

        ip_adapter = IPAdapterPlusXL(
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
            num_tokens=16,
        )

        base_inpainter = StableDiffusionXLInpaintPipeline.from_pretrained(
            params.model.inpainter,
            text_encoder=base.text_encoder,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            scheduler=base.scheduler,
            **cls.settings.loading_kwargs,
        ).to(rank, params.dtype)

        inpainter = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            params.model.inpainter,
            controlnet=ControlNetModel.from_pretrained(params.model.controlnet),
            **base_inpainter.components,
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

        xl_compel = Compel(
            [pipe.tokenizer, pipe.tokenizer_2],
            [pipe.text_encoder, pipe.text_encoder_2],
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
            base_inpainter=base_inpainter,
            inpainter=inpainter,
            compels=Compels(
                xl=xl_compel, refiner=refiner_compel, inpainter=inpainter_compel
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
        instance.outpaint_bases
        outpainted = instance.outpaint()
        smooth_edges = instance._smooth_edges(outpainted)
        redo_background = instance._redo_background(smooth_edges)
        final_refine = instance._final_refine(redo_background)
        return [
            instance.request.generation.images[0],
            # instance.faces[0],
            instance.tmp[0],  # face mask
            instance.tmp[1],  # face mask
            instance.frames[0],
            # instance.tmp[2].convert("RGB"),  # masked face
            # instance.controls[0].convert("RGB"),
            # instance.backgrounds[0],
            # ,
            instance.outpaint_bases[0],
            instance.masks[0].convert("RGB"),
            instance.og_masks[0].convert("RGB"),
            instance.edge_masks[0].convert("RGB"),
            # #
            outpainted[0],
            smooth_edges[0],
            redo_background[0],
            final_refine[0],
        ]


@dataclass
class ModelInstance:
    model: Model
    request: "ProcessRequest"
    tmp: list[Image.Image] = field(default_factory=list)

    def face_helper(self, images: list[Image.Image], tag: str = ""):
        return FaceHelper(
            self.params,
            self.models.face,
            images,
            logger=self.model.logger.bind(tag=tag),
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
                # control_guidance_end=self.params.conditioning_factor,
                scale=random.triangular(*self.params.conditioning_strength),
                **self.details_prompts.kwargs_for_xl(idx),
            )[0]
            for idx, img in enumerate(self.request.generation.images)
        ]

    def _touch_up_eyes(self, images: list[Image.Image]) -> list[Image.Image]:
        face_helper = self.face_helper(images, "eyes")
        masks, colors = map(list, zip(*face_helper.eye_masks()))
        self.model.logger.info("Colors: {}", colors)
        return [
            self.models.base_inpainter(
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

    @cached_property
    @collect
    def frames(self) -> Generator[Image.Image, None, None]:
        for idx, face in self.face_helper(self.faces, "faces").primary_face_bounds():
            self.tmp.append(draw_masks(self.faces[idx], [face]))
            bounds = Bounds.from_face(self.dims, face)
            self.model.logger.info("Face {} Bounds: {}", idx, bounds)
            target_percent = random.triangular(0.70, 0.75)

            curr_width, curr_height = bounds.size()
            target_width, target_height = [int(x * target_percent) for x in self.dims]
            d_w, d_h = target_width - curr_width, target_height - curr_height
            delta = max(d_w, d_h)
            padding = delta / max(self.dims)
            self.model.logger.info("Length Delta: {}, Padding: {}", delta, padding)

            face_img = np.asarray(self.faces[idx])[
                bounds.slice(self.params.mask_padding)
            ]
            self.tmp.append(ImageOps.pad(to_pil_image(face_img), self.dims))
            embed = np.asarray(to_pil_image(face_img).resize(bounds.size(padding)))

            frame = np.zeros((*self.dims, 3), dtype=np.uint8)
            frame[bounds.slice(padding)] = embed
            yield to_pil_image(frame)

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
    def controls(self) -> Generator[Image.Image, None, None]:
        yield from map(itemgetter(2), self._outpaint_bases)

    @cached_property
    @collect
    def _masks(self) -> Generator[tuple[Image.Image, Image.Image], None, None]:
        yield from map(itemgetter(1), self._outpaint_bases)

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

    def _base_background(self, idx: int) -> Image.Image:
        return self.models.base(
            generator=self.generator,
            guidance_scale=self.params.guidance_scale,
            num_inference_steps=self.params.inpainting_steps,
            **self.background_prompts.kwargs_for_xl(idx),
        ).images[0]

    @cached_property
    def backgrounds(self) -> list[Image.Image]:
        return [self._base_background(idx) for idx in range(len(self.frames))]

    @cached_property
    @collect
    def _outpaint_bases(
        self,
    ) -> Generator[
        tuple[Image.Image, tuple[Image.Image, Image.Image], Image.Image], None, None
    ]:
        self.model.logger.info("BG HELPER:")
        bg_face_helper = self.face_helper(self.backgrounds, "backgrounds")
        bg_face_bounds = bg_face_helper.primary_face_bounds()

        self.model.logger.info("FRAME HELPER:")
        frame_face_helper = self.face_helper(self.frames, "frames")
        frame_face_bounds = frame_face_helper.primary_face_bounds()

        for idx, frame_img in enumerate(self.frames):
            bg = self.backgrounds[idx]
            bg_face = bg_face_bounds[idx][1]
            bg_bounds = Bounds.from_face(self.dims, bg_face)

            frame = np.asarray(frame_img)
            frame_face = frame_face_bounds[idx][1]
            frame_bounds = Bounds.from_face(self.dims, frame_face)
            frame_mask, frame_og_mask = self._get_masks(frame, frame_face)

            frame_mask = np.asarray(frame_mask.convert("RGB"))
            frame_og_mask = np.asarray(frame_og_mask.convert("RGB"))

            slice = (frame_mask == 0) & (frame != 0)
            masked_face = np.where(slice, frame, np.asarray(bg))

            self.model.logger.info(
                "Using bg_bounds: {}, fg_bounds: {}", bg_bounds, frame_bounds
            )

            (Cx, Cy) = bg_bounds.center
            (Sx, Sy) = frame_bounds.size()
            start_x, start_y = max(Cx - Sx // 2, 0), max(Cy - Sy // 2, 0)
            end_x, end_y = min(Cx + Sx // 2, bg.width - 1), min(
                Cy + Sy // 2, bg.height - 1
            )

            if end_x - start_x != Sx:
                if start_x == 0:
                    end_x = Sx
                else:
                    start_x = end_x - Sx
            if end_y - start_y != Sy:
                if start_y == 0:
                    end_y = Sy
                else:
                    start_y = end_y - Sy

            self.model.logger.info(
                "{}",
                {
                    "idx": idx,
                    "bg": (Cx, Cy),
                    "fg": (Sx, Sy),
                    "start": (start_x, start_y),
                    "end": (end_x, end_y),
                    "bounds": frame_bounds.slice(),
                },
            )

            bg_image = np.array(bg)
            bg_image[start_y:end_y, start_x:end_x] = masked_face[frame_bounds.slice()]

            mask = np.full(bg_image.shape, 255, dtype=np.uint8)
            mask[start_y:end_y, start_x:end_x] = frame_mask[frame_bounds.slice()]

            og_mask = np.full(bg_image.shape, 255, dtype=np.uint8)
            og_mask[start_y:end_y, start_x:end_x] = frame_og_mask[frame_bounds.slice()]

            control_data = np.asarray(
                self.request.generation.controls[idx]
                .resize(frame_bounds.size())
                .convert("RGB")
            )
            control = np.zeros(bg_image.shape, dtype=np.uint8)
            control[start_y:end_y, start_x:end_x] = control_data

            yield (
                to_pil_image(bg_image, mode="RGB"),
                (
                    to_pil_image(mask, mode="RGB").convert("L"),
                    to_pil_image(og_mask, mode="RGB").convert("L"),
                ),
                to_pil_image(control, mode="RGB").convert("L"),
            )

    @property
    def outpaint_bases(self) -> list[Image.Image]:
        return list(map(itemgetter(0), self._outpaint_bases))

    @collect
    def outpaint(self) -> Generator[Image.Image, None, None]:
        latents = [
            self.models.inpainter(
                image=image,
                mask_image=self.masks[idx],
                control_image=self.controls[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                output_type="latent",
                denoising_end=self.params.high_noise_frac,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(self.outpaint_bases)
        ]

        refined = [
            self.models.refiner(
                image=latent,
                latents=latent,
                mask_image=self.og_masks[idx],
                generator=self.generator,
                strength=0.85,
                denoising_start=self.params.high_noise_frac,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.steps,
                **self.details_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, latent in enumerate(latents)
        ]

        for idx, img in enumerate(refined):
            img = np.array(img)
            slice = np.asarray(self.masks[idx].convert("RGB")) == 0
            img[slice] = np.asarray(self.outpaint_bases[idx])[slice]
            yield to_pil_image(img)

    def _smooth_edges(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.refiner(
                image=img,
                mask_image=self.edge_masks[idx],
                generator=self.generator,
                strength=0.25,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    def _redo_background(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.inpainter(
                image=image,
                mask_image=self.masks[idx],
                control_image=self.controls[idx],
                generator=self.generator,
                strength=0.99,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.details_prompts.kwargs_for_inpainter(idx),
            ).images[0]
            for idx, image in enumerate(images)
        ]

    def _final_refine(self, images: list[Image.Image]) -> list[Image.Image]:
        return [
            self.models.refiner(
                image=img,
                mask_image=self.edge_masks[idx],
                generator=self.generator,
                strength=0.25,
                guidance_scale=self.params.guidance_scale,
                num_inference_steps=self.params.inpainting_steps,
                **self.merge_prompts.kwargs_for_refiner(idx),
            ).images[0]
            for idx, img in enumerate(images)
        ]

    def generate(self) -> list[Image.Image]:
        outpainted = self.outpaint()
        smooth_edges = self._smooth_edges(outpainted)
        redo_background = self._redo_background(smooth_edges)
        return self._final_refine(redo_background)
