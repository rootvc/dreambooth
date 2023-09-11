import os
import random
from collections import Counter
from contextlib import ExitStack
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import numpy as np
from cachetools import Cache, cachedmethod
from controlnet_aux import LineartDetector
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    T2IAdapter,
)
from diffusers.loaders import LoraLoaderMixin
from modal import Volume, method

from dreambooth.train.helpers.face import FaceHelper
from one_shot.ensemble import StableDiffusionXLAdapterEnsemblePipeline
from one_shot.params import Params, Settings
from one_shot.prompt import Prompts
from one_shot.types import M
from one_shot.utils import collect, images, open_image, set_torch_config


class OneShotDreambooth:
    settings = Settings()

    def __init__(self, volume: Volume):
        self.volume = volume
        self.params = Params()
        self.exit_stack = ExitStack()
        self.dirty = False

    def __enter__(self):
        set_torch_config()
        self._load_models()
        self._set_cache_monitor()

    def __exit__(self):
        self.exit_stack.close()
        if self.dirty:
            self.volume.commit()

    @method()
    def warm(self):
        return Request(self, "test").generate()

    @method()
    def generate(self, id: str):
        return Request(self, id).generate()

    def _load_models(self):
        self.detector = self._download_model(
            LineartDetector, self.params.model.detector
        )

        pipe = self.ensemble = self._download_model(
            StableDiffusionXLAdapterEnsemblePipeline,
            self.params.model.name,
            refiner=self._download_model(DiffusionPipeline, self.params.model.refiner),
            adapter=self._download_model(T2IAdapter, self.params.model.t2i_adapter),
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        for repo, lora in self.params.model.loras.items():
            weights = self._download_model(
                LoraLoaderMixin, repo, weight_name=lora, method="lora_state_dict"
            )
            pipe.load_lora_weights(weights)

        pipe.fuse_lora(lora_scale=self.params.lora_scale)

    def _set_cache_monitor(self):
        cache_dir = Path(os.environ["CACHE_DIR"])
        mtime = cache_dir.stat().st_mtime
        self.exit_stack.callback(
            lambda: setattr(
                self,
                "dirty",
                True if cache_dir.stat().st_mtime != mtime else self.dirty,
            )
        )

    def _download_model(
        self, klass: type[M], name: str, method: str = "from_pretrained", **kwargs
    ) -> M:
        meth = getattr(klass, method)
        try:
            return meth(name, **kwargs, local_files_only=True)
        except OSError:
            self.dirty = True
        return meth(name, **kwargs).to("cuda", dtype=self.params.dtype)


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

    @cachedmethod(Cache)
    @collect
    def images(self) -> Generator[np.ndarray, None, None]:
        for path in images(self.image_dir):
            if path not in self._images:
                yield open_image(self.params, path)

    @cachedmethod(Cache)
    @collect
    def controls(self):
        for image in self.images():
            yield self.detector(
                image,
                detect_resolution=self.params.detect_resolution,
                image_resolution=self.params.model.resolution,
            )

    def demographics(self):
        res = [FaceHelper(self.params, img).demographics() for img in self.images()]
        return {k: Counter(r[k] for r in res).most_common(1)[0] for k in res[0]}

    def generate(self):
        return self.ensemble(
            image=random.choices(list(self._controls()), k=self.params.images),
            prompts=Prompts(
                self.ensemble,
                [
                    self.params.prompt_template.format(prompt=p, **self._demographics())
                    for p in random.sample(self.params.prompts, k=self.params.images)
                ],
                self.params.negative_prompt,
            ),
            guidance_scale=self.params.guidance_scale,
            adapter_conditioning_scale=self.params.conditioning_strength,
            adapter_conditioning_factor=self.params.conditioning_factor,
            high_noise_frac=self.params.high_noise_frac,
        ).images
