import os
import random
from contextlib import ExitStack
from functools import cache, cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import torch
from accelerate.utils import get_max_memory
from controlnet_aux import LineartDetector
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    T2IAdapter,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import AttnProcessor2_0
from dreambooth_old.train.download_eval_models import download as download_eval_models
from dreambooth_old.train.shared import grid
from loguru import logger
from modal import Volume
from PIL import Image
from torchvision.transforms.functional import to_tensor

from one_shot.ensemble import StableDiffusionXLAdapterEnsemblePipeline
from one_shot.face import Face
from one_shot.params import Params, Settings
from one_shot.prompt import Prompts
from one_shot.types import M
from one_shot.utils import (
    close_all_files,
    collect,
    get_mtime,
    images,
    open_image,
)


class OneShotDreambooth:
    settings = Settings()

    def __init__(self, volume: Volume):
        self.volume = volume
        self.params = Params()
        self.exit_stack = ExitStack()
        self.dirty = False

    def __enter__(self):
        logger.info("Starting...")
        self._load_models()
        logger.info("Loaded models")
        self._set_cache_monitor()
        return self

    def __exit__(self, *_args):
        logger.info("Exiting...")
        self.exit_stack.close()
        if self.dirty:
            logger.warning("Cache was modified, committing")
            del self.ensemble, self.detector
            close_all_files(os.environ["CACHE_DIR"])
            self.volume.commit()

    def _load_models(self):
        if download_eval_models(Path(self.settings.cache_dir)):
            self.dirty = True

        Face.load_models()

        self.detector = self._download_model(
            LineartDetector, self.params.model.detector, default_kwargs={}
        ).to("cuda")

        refiner = self._download_model(DiffusionPipeline, self.params.model.refiner)
        refiner.unet.set_attn_processor(AttnProcessor2_0())
        refiner.unet = torch.compile(
            refiner.unet.to(memory_format=torch.channels_last),
            fullgraph=True,
        )

        pipe = self.ensemble = self._download_model(
            StableDiffusionXLAdapterEnsemblePipeline,
            self.params.model.name,
            refiner=refiner,
            adapter=self._download_model(
                T2IAdapter,
                self.params.model.t2i_adapter,
            ),
        )

        pipe.unet.set_attn_processor(AttnProcessor2_0())
        pipe.unet = torch.compile(
            pipe.unet.to(memory_format=torch.channels_last),
            fullgraph=True,
        )

        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        for repo, lora in self.params.model.loras.items():
            self._download_model(
                LoraLoaderMixin, repo, weight_name=lora, method="lora_state_dict"
            )
            pipe.load_lora_weights(repo, weight_name=lora)

        pipe.fuse_lora(lora_scale=self.params.lora_scale)

    def _set_cache_monitor(self):
        mtime = get_mtime(Path(os.environ["CACHE_DIR"]))
        self.exit_stack.callback(
            lambda: setattr(
                self,
                "dirty",
                True
                if get_mtime(Path(os.environ["CACHE_DIR"])) > mtime
                else self.dirty,
            )
        )

    def _download_model(
        self,
        klass: type[M],
        name: str,
        method: str = "from_pretrained",
        default_kwargs: dict = {
            "local_files_only": True,
            "use_safetensors": True,
            "device_map": "auto",
            "max_memory": {
                k: v
                for k, v in get_max_memory().items()
                if k != torch.cuda.device_count() - 1
            },
            "low_cpu_mem_usage": True,
        },
        **kwargs,
    ) -> M:
        logger.info(f"Loading {klass.__name__}({name})...")

        meth = getattr(klass, method)
        try:
            return meth(name, **default_kwargs, **kwargs)
        except OSError:
            self.dirty = True
        logger.warning(f"Downloading {klass.__name__}({name})...")
        model = meth(
            name,
            **{k: v for k, v in default_kwargs.items() if k != "local_files_only"},
            **kwargs,
        )

        if hasattr(model, "to"):
            return model.to(torch_dtype=self.params.dtype)
        elif isinstance(model, dict):
            for k, v in model.items():
                model[k] = v.to(dtype=self.params.dtype)
        return model


class Request:
    ensemble: StableDiffusionXLAdapterEnsemblePipeline

    def __init__(self, dreambooth: OneShotDreambooth, id: str):
        self.dreambooth = dreambooth
        self.id = id

    def __getattr__(self, attr: str):
        return getattr(self.dreambooth, attr)

    @cached_property
    def image_dir(self):
        dir = Path(self.exit_stack.enter_context(TemporaryDirectory()))
        (self.settings.bucket / "dataset" / self.id).download_to(dir)
        return dir

    @cache
    @collect
    def images(self) -> Generator[Image.Image, None, None]:
        logger.info("Loading images...")
        for path in images(self.image_dir):
            logger.debug(f"Loading {path}...")
            yield open_image(self.params, path)

    @cache
    @collect
    def controls(self):
        logger.info("Loading controls...")
        for i, image in enumerate(self.face.faces()):
            logger.debug(f"Loading controls for {i}...")
            yield self.detector(
                image,
                detect_resolution=self.params.detect_resolution,
                image_resolution=self.params.model.resolution,
            )

    @cached_property
    def face(self):
        return Face(self.params, self.images())

    def demographics(self):
        return {"race": "beautiful", "gender": "person"}
        return self.face.demographics()

    @torch.inference_mode()
    def generate(self):
        logger.info("Generating...")
        return grid(self.face.faces())
        images = random.choices(list(self.controls()), k=self.params.images)
        images = self.ensemble(
            image=torch.stack([to_tensor(i) for i in images]).to("cuda"),
            prompts=Prompts(
                self.ensemble,
                [
                    self.params.prompt_template.format(prompt=p, **self.demographics())
                    for p in random.sample(self.params.prompts, k=self.params.images)
                ],
                self.params.negative_prompt,
            ),
            guidance_scale=self.params.guidance_scale,
            adapter_conditioning_scale=self.params.conditioning_strength,
            adapter_conditioning_factor=self.params.conditioning_factor,
            high_noise_frac=self.params.high_noise_frac,
        ).images
        return grid(images)
