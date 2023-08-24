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
from diffusers.models.vae import DiagonalGaussianDistribution
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import (
    CLIPTokenizer,
)

from dreambooth.params import Class, HyperParams
from dreambooth.train.accelerators import BaseAccelerator
from dreambooth.train.sdxl.utils import tokenize_prompt
from dreambooth.train.shared import (
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
        images = (
            torch.stack(images)
            .to(self.accelerator.device, memory_format=torch.contiguous_format)
            .float()
        )
        latent_dist = self.vae.to(self.accelerator.device).encode(images).latent_dist
        latent_dist = DiagonalGaussianDistribution(latent_dist.parameters.to("cpu"))

        tokens = [
            (torch.cat([i1, p1], dim=0), torch.cat([i2, p2], dim=0))
            for ((i1, i2), (p1, p2)) in zip(
                map(itemgetter("instance_tokens"), batch),
                map(itemgetter("prior_tokens"), batch),
            )
        ]

        return {
            "latent_dist": latent_dist,
            "tokens": tokens,
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
        tokenizers: tuple[CLIPTokenizer, CLIPTokenizer],
        size: int,
        vae_scale_factor: float,
        augment: bool = True,
    ):
        self.instance = instance
        self.prior = prior
        self.size = size
        self.vae_scale_factor = vae_scale_factor
        self.tokenizers = tokenizers
        self.augment = augment

        self._length = max(len(self.instance_images), len(self.prior_images))

    @cached_property
    def prior_images(self):
        path = self.prior.data
        return images(path)

    @cached_property
    def instance_images(self):
        path = self.instance.data
        return images(path)

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
            return exif_transpose(img)
        else:
            return exif_transpose(img.convert("RGB"))

    def _instance_image(self, index):
        path = self.instance_images[index % len(self.instance_images)]
        do_augment, index = divmod(index, len(self.instance_images))
        image = self.image_transforms(do_augment)(self.open_image(path))
        prompt = self.instance.prompt

        return {
            "instance_image": image,
            "instance_tokens": (
                tokenize_prompt(self.tokenizers[0], prompt),
                tokenize_prompt(self.tokenizers[1], prompt),
            ),
        }

    def _prior_image(self, index):
        path = self.prior_images[index % len(self.prior_images)]
        image = self.image_transforms(False)(self.open_image(path))
        prompt = self.prior.prompt

        return {
            "prior_image": image,
            "prior_tokens": (
                tokenize_prompt(self.tokenizers[0], prompt),
                tokenize_prompt(self.tokenizers[1], prompt),
            ),
        }

    def __getitem__(self, index):
        return {**self._instance_image(index), **self._prior_image(index)}
