import itertools
from functools import cached_property
from operator import itemgetter
from pathlib import Path

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.config_utils
import torch._functorch.config
import torch._inductor.config
import torch.backends.cuda
import torch.backends.cudnn
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors
import torch.jit
from diffusers import (
    AutoencoderKL,
)
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    CLIPTokenizer,
)

from dreambooth.params import Class, HyperParams
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.shared import (
    depth_image_path,
    depth_transforms,
    image_transforms,
    images,
)


class PromptDataset(Dataset):
    def __init__(self, prompt: str, n: int):
        self.prompt = prompt
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        return {"prompt": self.prompt, "index": i}


class WarmedCachedLatentsDataset(Dataset):
    def __init__(self, cached: dict[int, dict]):
        self.cached = cached

    def __getitem__(self, index):
        return self.cached[index]

    def __len__(self):
        return len(self.cached)


class CachedLatentsDataset(Dataset):
    def __init__(
        self,
        accelerator: BaseAccelerator,
        dataset: Dataset,
        params: HyperParams,
        vae: AutoencoderKL,
    ):
        self.dataset = dataset
        self.params = params
        self.vae = vae
        self.accelerator = accelerator
        self._length = len(self.dataset)
        self._cached_latents = {}
        self._warmed = False

    @torch.no_grad()
    def _compute_latents(self, batch: list[dict[str, torch.FloatTensor]]):
        images = list(
            itertools.chain(
                map(itemgetter("instance_image"), batch),
                map(itemgetter("prior_image"), batch),
            )
        )
        images = torch.stack(images).to(
            self.accelerator.device,
            dtype=self.params.dtype,
            memory_format=torch.contiguous_format,
        )

        # depth_images = list(
        #     itertools.chain(
        #         map(itemgetter("instance_depth_image"), batch),
        #         map(itemgetter("prior_depth_image"), batch),
        #     )
        # )
        # depth_images = (
        #     torch.stack(depth_images)
        #     .float()
        #     .to(
        #         self.accelerator.device,
        #         memory_format=torch.contiguous_format,
        #     )
        # )

        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.squeeze(0).to(self.accelerator.device).float()

        input_ids = list(
            itertools.chain(
                map(itemgetter("instance_prompt_ids"), batch),
                map(itemgetter("prior_prompt_ids"), batch),
            )
        )
        input_ids = torch.cat(input_ids, dim=0).to(
            self.accelerator.device, dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "latents": latents,
            # "depth_values": depth_images,
        }

    def __len__(self):
        return self._length // self.params.batch_size

    def __getitem__(self, i: int):
        if not self._warmed and i not in self._cached_latents:
            s = self.params.batch_size
            self._cached_latents[i] = self._compute_latents(
                [self.dataset[idx] for idx in range(s * i, s * (i + 1))]
            )
        return self._cached_latents[i]

    def warm(self):
        for i in tqdm(range(len(self)), disable=not self.accelerator.is_main_process):
            self[i]
        self._warmed = True
        return WarmedCachedLatentsDataset(self._cached_latents)


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        *,
        instance: Class,
        prior: Class,
        tokenizer: CLIPTokenizer,
        size: int,
        vae_scale_factor: float,
        augment: bool = True,
    ):
        self.instance = instance
        self.prior = prior
        self.size = size
        self.vae_scale_factor = vae_scale_factor
        self.tokenizer = tokenizer
        self.augment = augment

        self._length = max(len(self.instance_images), len(self.prior_images))

    @staticmethod
    def depth_image_path(path):
        return depth_image_path(path)

    @cached_property
    def prior_images(self):
        path = self.prior.data
        return images(path)

    @cached_property
    def instance_images(self):
        path = self.instance.data
        return images(path)

    def depth_transform(self):
        return depth_transforms(self.size, self.vae_scale_factor)

    def image_transforms(self, augment: bool = True):
        return image_transforms(self.size, self.augment and augment)

    def __len__(self):
        return self._length

    def __iter__(self):
        return (self[i] for i in range(self._length))

    @staticmethod
    def open_image(path: Path, convert: bool = True):
        img = Image.open(path)
        if not convert or img.mode == "RGB":
            return img
        else:
            return img.convert("RGB")

    def _instance_image(self, index):
        path = self.instance_images[index % len(self.instance_images)]
        do_augment, index = divmod(index, len(self.instance_images))
        image = self.image_transforms(do_augment)(self.open_image(path))

        # depth_path = self.depth_image_path(path)
        # depth_image = self.depth_transform()(self.open_image(depth_path, convert=False))

        return {
            "instance_image": image,
            # "instance_depth_image": depth_image,
            "instance_prompt_ids": self.tokenizer(
                self.instance.prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
        }

    def _prior_image(self, index):
        path = self.prior_images[index % len(self.prior_images)]
        image = self.image_transforms(False)(self.open_image(path))

        # depth_path = self.depth_image_path(path)
        # depth_image = self.depth_transform()(self.open_image(depth_path, convert=False))

        return {
            "prior_image": image,
            # "prior_depth_image": depth_image,
            "prior_prompt_ids": self.tokenizer(
                self.prior.prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids,
        }

    def __getitem__(self, index):
        return {**self._instance_image(index), **self._prior_image(index)}
