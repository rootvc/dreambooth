import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing
from controlnet_aux import LineartDetector
from diffusers import (
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    T2IAdapter,
)
from diffusers.loaders import LoraLoaderMixin
from dreambooth_old.train.download_eval_models import download as download_eval_models
from loguru import logger
from modal import Volume
from PIL import Image

from one_shot.face import Face
from one_shot.params import Params, Settings
from one_shot.types import M
from one_shot.utils import (
    civitai_path,
    close_all_files,
    consolidate_cache,
    download_civitai_model,
    get_mtime,
)

from .process import Process, ProcessRequest

mp = torch.multiprocessing.get_context("forkserver")


@dataclass
class SharedModels:
    detector: LineartDetector


@dataclass
class Queues:
    proc: list["torch.multiprocessing.Queue[Optional[ProcessRequest]]"]
    response: "torch.multiprocessing.Queue[list[Image.Image]]"


class OneShotDreambooth:
    settings = Settings()
    world_size = torch.cuda.device_count() - 1

    models: SharedModels
    queues: Queues
    ctx: torch.multiprocessing.ProcessContext

    def __init__(self, volume: Volume):
        self.volume = volume
        self.params = Params()
        self.exit_stack = ExitStack()
        self.dirty = False

    def __enter__(self, skip_procs: bool = False):
        logger.info("Starting with max memory: {}", self.settings.max_memory())
        if not skip_procs:
            self.queues, self.ctx = self._start_processes()
        self._download_models()
        torch.cuda.empty_cache()
        self.models = self._load_models()
        self._set_cache_monitor()
        return self

    def __exit__(self, *_args):
        logger.info("Exiting...")
        consolidate_cache(os.environ["CACHE_DIR"], os.environ["CACHE_STAGING_DIR"])
        self.exit_stack.close()
        if self.dirty:
            logger.warning("Cache was modified, committing")
            self._cleanup()
            close_all_files(os.environ["CACHE_DIR"])
            self.volume.commit()
            logger.warning("Cache committed")
        elif hasattr(self, "ctx"):
            self.ctx.join()

    def _cleanup(self):
        del self.models
        if not hasattr(self, "ctx"):
            return
        for proc in self.ctx.processes:
            proc.terminate()
            proc.join()
            proc.close()
        self.queues.response.close()
        for queue in self.queues.proc:
            queue.close()
        del self.ctx, self.queues

    @torch.inference_mode()
    def _download_models(self):
        logger.info("Downloading models...")
        torch.cuda.set_device(torch.cuda.device_count() - 1)

        if download_eval_models(Path(self.settings.cache_dir)):
            self.dirty = True
        Face.load_models()
        self._download_model(
            StableDiffusionXLImg2ImgPipeline, self.params.model.refiner
        )
        self._download_model(
            StableDiffusionXLAdapterPipeline,
            self.params.model.name,
            adapter=self._download_model(
                T2IAdapter, self.params.model.t2i_adapter, method="from_pretrained"
            ),
        )
        self._download_model(
            StableDiffusionXLInpaintPipeline,
            self.params.model.inpainter,
        )
        for loras in self.params.model.loras.values():
            for repo, lora in loras.items():
                if lora == "civitai":
                    if civitai_path(repo).exists():
                        continue
                    self.dirty = True
                    download_civitai_model(repo)
                else:
                    self._download_model(
                        LoraLoaderMixin,
                        repo,
                        weight_name=lora,
                        method="lora_state_dict",
                    )

    @torch.inference_mode()
    def _load_models(self) -> SharedModels:
        logger.info("Loading models...")
        detector = self._download_model(
            LineartDetector,
            self.params.model.detector,
            default_kwargs={},
            method="from_pretrained",
        ).to(f"cuda:{torch.cuda.device_count() - 1}")
        return SharedModels(detector=detector)

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
        method: str = "download",
        default_kwargs: dict = {
            "device_map": "auto",
            "max_memory": settings.max_memory(torch.cuda.device_count() - 1),
            **settings.loading_kwargs,
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

    def _start_processes(self) -> tuple[Queues, torch.multiprocessing.ProcessContext]:
        logger.info("Starting processes...")
        queues = Queues(
            proc=[mp.Queue(2) for _ in range(self.world_size)],
            response=mp.Queue(self.world_size),
        )

        ctx = torch.multiprocessing.start_processes(
            Process.run,
            args=(self.world_size, self.params, queues),
            nprocs=self.world_size,
            join=False,
            start_method="forkserver",
        )

        def cleanup():
            logger.info("Stopping processes...")
            for queue in queues.proc:
                queue.put_nowait(None)

        self.exit_stack.callback(cleanup)
        return queues, ctx
