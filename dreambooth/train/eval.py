import hashlib
import itertools
import random
import shutil
from datetime import timedelta
from functools import cached_property
from typing import cast

import cv2
import torch
import torch.distributed
import tqdm
import wandb
from compel import Compel, ReturnedEmbeddingsType
from diffusers import (
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
)
from PIL import Image
from rich import print
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import make_grid

from dreambooth.params import Class, HyperParams
from dreambooth.train.helpers.face import FaceHelper
from dreambooth.train.shared import (
    dprint,
    is_main,
    main_process_only,
)
from dreambooth.train.shared import (
    images as get_images,
)


class PromptDataset(Dataset):
    def __init__(
        self,
        params: HyperParams,
        pipe: StableDiffusionXLControlNetPipeline | StableDiffusionControlNetPipeline,
        n: int,
    ):
        self.params = params
        if isinstance(pipe, StableDiffusionXLControlNetPipeline):
            self.compel = Compel(
                [pipe.tokenizer, pipe.tokenizer_2],
                [pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=pipe.device,
            )
        else:
            self.compel = Compel(
                pipe.tokenizer,
                pipe.text_encoder,
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NORMALIZED,
                device=pipe.device,
            )
        self.n = n
        self.prompts = random.sample(
            [(p, self.compel([p])) for p in self.params.eval_prompts],
            k=n,
        )

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return [self.prompts[i]]


class Evaluator:
    def __init__(
        self,
        device: torch.device,
        params: HyperParams,
        instance_class: Class,
        pipeline: StableDiffusionControlNetPipeline,
    ):
        self.params = params
        self.instance_class = instance_class
        self.device = device
        self.pipeline = pipeline

    def _preprocess(self, pil_image: Image.Image):
        return FaceHelper.preprocess(self.params, pil_image)

    @cached_property
    def test_images(self):
        return [
            Image.fromarray(FaceHelper(self.params, Image.open(p)).mask())
            for p in get_images(self.instance_class.data)
        ]

    @cached_property
    def cond_images(self):
        dprint("Preprocessing images...")
        images = [
            self._preprocess(Image.open(p))
            for p in get_images(self.instance_class.data)
        ]
        print("Done preprocessing...")
        return images

    @cached_property
    def refiner(self) -> tuple[StableDiffusionXLImg2ImgPipeline, Compel]:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.params.model.refiner,
            torch_dtype=self.params.dtype,
            variant=self.params.model.variant,
        ).to(self.device)
        refiner.enable_xformers_memory_efficient_attention()

        compel = Compel(
            tokenizer=refiner.tokenizer_2,
            text_encoder=refiner.text_encoder_2,
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=True,
            device=refiner.device,
        )

        return refiner, compel

    def _gen_latents(self, ds: PromptDataset, batch: list[tuple[str, torch.Tensor]]):
        prompts, embeddings = zip(*batch)
        # conditionings, pools = zip(*embeddings)

        # embeds = torch.cat(conditionings, dim=0)
        # pooled_embeds = torch.cat(pools, dim=0)
        # neg_embeds, neg_pools = ds.compel([self.params.negative_prompt] * len(prompts))

        # [
        #     embeds,
        #     neg_embeds,
        # ] = ds.compel.pad_conditioning_tensors_to_same_length([embeds, neg_embeds])

        # (embeds, pooled_embeds, neg_embeds, neg_pools) = [
        #     x.to(self.device, dtype=self.params.dtype)
        #     for x in (embeds, pooled_embeds, neg_embeds, neg_pools)
        # ]

        return self.pipeline(
            prompt=list(prompts),
            negative_prompt=[self.params.negative_prompt] * len(prompts),
            # prompt_embeds=embeds,
            # pooled_prompt_embeds=pooled_embeds,
            # negative_prompt_embeds=neg_embeds,
            # negative_pooled_prompt_embeds=neg_pools,
            # width=self.params.model.resolution,
            # height=self.params.model.resolution,
            # image=random.choices(self.test_images, k=len(prompts)),
            num_inference_steps=self.params.test_steps,
            guidance_scale=self.params.test_guidance_scale,
            # strength=self.params.test_strength,
            # controlnet_conditioning_scale=self.params.test_strength,
            cross_attention_kwargs={"scale": self.params.lora_alpha},
        ).images

    def _gen_images(self) -> list[tuple[str, Image.Image]]:
        ds = PromptDataset(self.params, self.pipeline, n=self.params.test_images)
        loader = DataLoader(
            ds,
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
            batch_size=self.params.test_images,
        )

        all_images = []
        for batch in tqdm.tqdm(loader):
            prompts, _ = zip(*batch)
            latents = self._gen_latents(ds, batch)
            # prompts, _ = zip(*batch)
            # refiner, compel = self.refiner
            # # embeds, pools = compel(list(prompts))
            # # neg_embeds, neg_pools = compel([self.params.negative_prompt] * len(prompts))
            # # [
            # #     embeds,
            # #     neg_embeds,
            # # ] = ds.compel.pad_conditioning_tensors_to_same_length([embeds, neg_embeds])
            # images = refiner(
            #     prompt=[
            #         p.replace(self.params.token, self.params.source_token)
            #         for p in prompts
            #     ],
            #     negative_prompt=[self.params.negative_prompt] * len(prompts),
            #     # prompt_embeds=embeds.to(dtype=self.params.dtype),
            #     # pooled_prompt_embeds=pools.to(dtype=self.params.dtype),
            #     # negative_prompt_embeds=neg_embeds.to(dtype=self.params.dtype),
            #     # negative_pooled_prompt_embeds=neg_pools.to(dtype=self.params.dtype),
            #     image=latents,
            # ).images
            all_images.extend(zip(prompts, latents))

        return all_images

    @main_process_only
    def _upload_images(self):
        if not self.params.debug_outputs:
            return
        wandb.run.log(
            {
                "source": [
                    wandb.Image(self.test_images[0], caption="image"),
                    wandb.Image(self.cond_images[0], caption="condition"),
                ]
            }
        )
        wandb.run.log(
            {
                "original": [
                    wandb.Image(
                        str(p),
                        caption=(self.params.image_output_path / "prompt" / p.stem)
                        .with_suffix(".txt")
                        .read_text(),
                    )
                    for p in (self.params.image_output_path / "original").glob("*.png")
                ]
            }
        )

    def _grid(self, images: list[Image.Image]) -> Image.Image:
        tensors = torch.stack([pil_to_tensor(img) for img in images])
        grid = make_grid(tensors, nrow=2, pad_value=255, padding=10)
        return to_pil_image(grid)

    def _paths(self):
        paths = [
            self.params.image_output_path / "prompt",
            self.params.image_output_path / "original",
            self.params.image_output_path / "restored",
        ]
        if is_main():
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(exist_ok=True)
        return paths

    def wait_for_everyone(self):
        if torch.distributed.is_initialized():
            dprint("Waiting for other processes to finish...")
            torch.distributed.barrier(async_op=True).wait(timeout=timedelta(seconds=30))

    @torch.no_grad()
    def generate(self):
        dprint("Generating images...")
        prompts, images = cast(
            tuple[tuple[str], tuple[Image.Image]], list(zip(*self._gen_images()))
        )
        prompt_path, original_path, restored_path = self._paths()

        dprint("Cleaning up...")
        del self.pipeline

        all_restored = itertools.repeat(None)

        dprint(f"Saving {len(images)} images...")
        for prompt, image in zip(prompts, images):
            dprint(prompt)
            slug = hashlib.md5(prompt.encode()).hexdigest()
            (prompt_path / f"{slug}.txt").write_text(prompt)
            image.save(original_path / f"{slug}.png")

            dprint("Restoring...")
            if (restored := next(all_restored)) is None:
                continue
            path = str(restored_path / f"{slug}.png")
            cv2.imwrite(path, restored)

        if not self.params.debug_outputs:
            dprint("Saving grid...")
            grid = self._grid(images)
            grid.save(self.params.image_output_path / "grid.png")

        dprint("Waiting for upload...")
        self._upload_images()
        dprint("Done!")
