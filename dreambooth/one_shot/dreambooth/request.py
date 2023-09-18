import itertools
import random
from functools import cache, cached_property
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Generator

import torch
from dreambooth_old.train.shared import grid
from loguru import logger
from PIL import Image

from one_shot.dreambooth.process import GenerationRequest, ProcessRequest
from one_shot.face import Face
from one_shot.utils import (
    collect,
    images,
    open_image,
)

if TYPE_CHECKING:
    from dreambooth.one_shot.dreambooth import OneShotDreambooth


class Request:
    def __init__(self, dreambooth: "OneShotDreambooth", id: str):
        self.dreambooth = dreambooth
        self.id = id

    @cached_property
    def image_dir(self):
        dir = Path(self.dreambooth.exit_stack.enter_context(TemporaryDirectory()))
        (self.dreambooth.settings.bucket / "dataset" / self.id).download_to(dir)
        return dir

    @cache
    @collect
    def images(self) -> Generator[Image.Image, None, None]:
        logger.info("Loading images...")
        for path in images(self.image_dir):
            logger.debug(f"Loading {path}...")
            yield open_image(self.dreambooth.params, path)

    @cache
    @collect
    def controls(self):
        logger.info("Loading controls...")
        for i, image in enumerate(self.images()):
            logger.debug(f"Loading controls for {i}...")
            yield self.dreambooth.models.detector(
                image,
                detect_resolution=self.dreambooth.params.detect_resolution,
                image_resolution=self.dreambooth.params.model.resolution,
            )

    @cached_property
    def face(self):
        return Face(self.dreambooth.params, self.images())

    @cached_property
    def demographics(self):
        demos = self.face.demographics()
        logger.info(demos)
        return demos

    @torch.inference_mode()
    @collect
    def _generate(self):
        images = random.choices(self.controls(), k=self.dreambooth.params.images)
        prompts = random.sample(
            self.dreambooth.params.prompts, k=self.dreambooth.params.images
        )
        generation = GenerationRequest(images, prompts)

        logger.info("Sending generation requests...")
        for i, queue in enumerate(self.dreambooth.queues.proc):
            logger.debug(f"Sending request to {i}...")
            queue.put_nowait(ProcessRequest(self.demographics, generation))

        logger.info("Receiving generation responses...")
        for i in range(self.dreambooth.world_size):
            logger.debug(f"Receiving response from {i}...")
            try:
                yield self.dreambooth.queues.response.get(timeout=90)
            except Empty:
                self.dreambooth.ctx.join(0)
                raise

    def generate(self):
        return grid(
            self.controls() + list(itertools.chain.from_iterable(self._generate()))
        )
