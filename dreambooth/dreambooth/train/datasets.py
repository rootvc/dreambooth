import itertools
from functools import cached_property, lru_cache
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
from dreambooth.train.helpers.face import FaceHelper
from dreambooth.train.sdxl.utils import tokenize_prompt
from dreambooth.train.shared import (
    image_transforms,
    images,
)


class DiagonalGaussianDistributionWithTo(DiagonalGaussianDistribution):
    def to(self, *args, **kwargs):
        return self.__class__(self.parameters.to(*args, **kwargs))


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

    @torch.inference_mode()
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
        latent_dist = DiagonalGaussianDistributionWithTo(
            latent_dist.parameters.to("cpu")
        )

        tokens = list(
            itertools.chain(
                map(itemgetter("instance_tokens"), batch),
                map(itemgetter("prior_tokens"), batch),
            )
        )

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
        tokenizers: list[CLIPTokenizer],
        params: HyperParams,
        vae_scale_factor: float,
        augment: bool = True,
    ):
        self.instance = instance
        self.prior = prior
        self.params = params
        self.size = params.model.resolution
        self.vae_scale_factor = vae_scale_factor
        self.use_priors = bool(params.prior_loss_weight)
        self.tokenizers = tokenizers
        self.augment = augment

        self._length = max(
            len(self.instance_images), min(len(self.prior_images), params.prior_samples)
        )

    @cached_property
    def prior_images(self):
        path = self.prior.data
        return images(path)

    @cached_property
    def instance_images(self):
        path = self.instance.data
        return images(path)

    @lru_cache
    def masked_instance_image(self, index):
        path = self.instance_images[index]
        image = self.open_image(path)
        return Image.fromarray(FaceHelper(self.params, image).mask())

    def image_transforms(self, augment: bool = True):
        return image_transforms(self.size, self.augment and augment)

    def __len__(self):
        return self._length

    def __iter__(self):
        return (self[i] for i in range(self._length))

    @staticmethod
    def open_image(path: Path, convert: bool = True, mask: bool = False):
        img = exif_transpose(Image.open(path))
        if convert and img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _instance_image(self, index):
        do_augment, index = divmod(index, len(self.instance_images))
        if do_augment:
            image = self.masked_instance_image(index % len(self.instance_images))
        else:
            path = self.instance_images[index % len(self.instance_images)]
            image = self.open_image(path)

        image = self.image_transforms(do_augment)(image)
        prompt = self.instance.prompt

        return {
            "instance_image": image,
            "instance_tokens": [tokenize_prompt(t, prompt) for t in self.tokenizers],
        }

    def _prior_image(self, index):
        if not self.use_priors:
            return {}
        path = self.prior_images[index % len(self.prior_images)]
        image = self.image_transforms(False)(self.open_image(path))
        prompt = self.prior.prompt

        return {
            "prior_image": image,
            "prior_tokens": [tokenize_prompt(t, prompt) for t in self.tokenizers],
        }

    def __getitem__(self, index):
        return {**self._instance_image(index), **self._prior_image(index)}
