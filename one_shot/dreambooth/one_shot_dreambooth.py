import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.multiprocessing
from controlnet_aux.lineart import LineartDetector
from diffusers import (
    AutoencoderKL,
    StableDiffusionPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
    T2IAdapter,
)
from diffusers.loaders import LoraLoaderMixin
from modal import Volume
from transformers import BlipForQuestionAnswering, BlipProcessor, SamModel, SamProcessor

from one_shot import logger
from one_shot.dreambooth.process.models.base import SharedModels
from one_shot.dreambooth.process.process import (
    Process,
    ProcessRequest,
    ProcessResponse,
    ProcessResponseSentinel,
)
from one_shot.face import FaceHelperModels
from one_shot.params import IPAdapterParams, Params, Settings, T2IAdapterParams
from one_shot.types import M
from one_shot.utils import (
    SelectableList,
    civitai_path,
    close_all_files,
    consolidate_cache,
    download_civitai_model,
    get_mtime,
    load_hf_file,
)

mp = torch.multiprocessing.get_context("forkserver")


@dataclass
class Queues:
    proc: list["torch.multiprocessing.Queue[Optional[ProcessRequest]]"]
    response: "torch.multiprocessing.Queue[ProcessResponse | ProcessResponseSentinel]"


class OneShotDreambooth:
    settings = Settings()
    world_size = torch.cuda.device_count()

    models: SharedModels
    queues: Queues
    ctx: torch.multiprocessing.ProcessContext

    def __init__(self, volume: Volume):
        self.volume = volume
        self.params = Params()
        self.t2i_params = T2IAdapterParams()
        self.ip_params = IPAdapterParams()
        self.exit_stack = ExitStack()
        self.dirty = False

    def __enter__(self, skip_procs: bool = False):
        logger.info(
            "Starting with max memory: {} on cloud {} ({})",
            self.settings.max_memory(),
            os.getenv("MODAL_CLOUD_PROVIDER", "unknown"),
            os.getenv("MODAL_REGION", "???"),
        )
        if skip_procs:
            self._download_models()
            torch.cuda.empty_cache()
            self.models = self._load_models()
        else:
            self.models = self._load_models()
            self.queues, self.ctx = self._start_processes()
        self._set_cache_monitor()
        return self

    def __exit__(self, *_args):
        logger.info("Exiting...")
        consolidate_cache(os.environ["CACHE_DIR"], os.environ["CACHE_STAGING_DIR"])
        self.exit_stack.close()
        if self.dirty and not hasattr(self, "ctx"):
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

        self._download_model(
            BlipProcessor,
            self.params.model.vqa,
            method="from_pretrained",
            default_kwargs={"local_files_only": True},
        )
        self._download_model(
            BlipForQuestionAnswering,
            self.params.model.vqa,
            method="from_pretrained",
            default_kwargs={
                "torch_dtype": torch.float16,
                "local_files_only": True,
            },
        )
        self._download_model(
            SamProcessor,
            self.params.model.sam,
            method="from_pretrained",
            default_kwargs={"local_files_only": True},
        )
        self._download_model(
            SamModel,
            self.params.model.sam,
            method="from_pretrained",
            default_kwargs={
                "torch_dtype": torch.float16,
                "local_files_only": True,
            },
        )
        self._download_model(
            StableDiffusionXLAdapterPipeline,
            self.params.model.name,
            vae=self._download_model(
                AutoencoderKL,
                self.t2i_params.model.vae,
                method="from_pretrained",
                variant=None,
            ),
            adapter=self._download_model(
                T2IAdapter,
                self.t2i_params.model.t2i_adapter,
                method="from_pretrained",
            ),
        )
        self._download_model(
            LineartDetector,
            self.t2i_params.model.detector,
            default_kwargs={},
            method="from_pretrained",
        ).to(f"cuda:{torch.cuda.device_count() - 1}")
        self._download_model(
            StableDiffusionXLPipeline,
            self.params.model.name,
            vae=self._download_model(
                AutoencoderKL,
                self.params.model.vae,
                method="from_pretrained",
                variant=None,
            ),
        )
        for _, file in self.ip_params.model.ip_adapter.files:
            self._download_hf_file(self.ip_params.model.ip_adapter.repo, file)
        self._download_model(
            StableDiffusionPipeline,
            self.ip_params.model.ip_adapter.base,
            method="from_pretrained",
            vae=self._download_model(
                AutoencoderKL,
                self.ip_params.model.ip_adapter.vae,
                method="from_pretrained",
                variant=None,
            ),
            variant=None,
        )
        self._download_model(
            StableDiffusionXLInpaintPipeline,
            self.params.model.inpainter,
            vae=self._download_model(
                AutoencoderKL,
                self.params.model.vae,
                method="from_pretrained",
                variant=None,
            ),
        )
        self._download_model(
            StableDiffusionXLInpaintPipeline,
            self.params.model.refiner,
            vae=self._download_model(
                AutoencoderKL,
                self.params.model.vae,
                method="from_pretrained",
                variant=None,
            ),
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
        return SharedModels(
            face=FaceHelperModels.load(self.params, torch.cuda.device_count() - 1),
        )

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

    def _download_hf_file(self, repo: str, file: str):
        try:
            return load_hf_file(repo, file, local_files_only=True)
        except OSError:
            self.dirty = True
            return load_hf_file(repo, file)

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
            return meth(name, **{**default_kwargs, **kwargs})
        except OSError:
            self.dirty = True
        logger.warning(f"Downloading {klass.__name__}({name})...")
        model = meth(
            name,
            **{
                **{k: v for k, v in default_kwargs.items() if k != "local_files_only"},
                **kwargs,
            },
        )

        if hasattr(model, "to"):
            return model.to(self.params.dtype)
        elif isinstance(model, dict):
            for k, v in model.items():
                model[k] = v.to(dtype=self.params.dtype)

        logger.info(f"Loaded {klass.__name__}({name})")
        return model

    def _start_processes(self) -> tuple[Queues, torch.multiprocessing.ProcessContext]:
        logger.info("Starting processes...")
        queues = Queues(
            proc=[mp.Queue(2) for _ in range(self.world_size)],
            response=mp.Queue(self.world_size),
        )

        ctx = torch.multiprocessing.start_processes(
            Process.run,
            args=(
                self.world_size,
                self.params,
                SelectableList([self.t2i_params, self.ip_params]),
                queues,
                logger,
            ),
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
