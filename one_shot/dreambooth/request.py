import itertools
import json
import random
from functools import cache, cached_property
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Generator, Optional

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image

from one_shot.dreambooth.process import (
    GenerationRequest,
    ProcessRequest,
    ProcessResponseSentinel,
)
from one_shot.face import FaceHelper
from one_shot.utils import (
    collect,
    grid,
    images,
    open_image,
)

if TYPE_CHECKING:
    from one_shot.dreambooth import OneShotDreambooth


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
        return FaceHelper(
            self.dreambooth.params, self.dreambooth.models.face, self.images()
        )

    @cached_property
    def demographics(self):
        demos = self.face.demographics()
        logger.info(demos)
        return demos

    def _generate(self, params: Optional[dict[str, list[int | float]]] = None):
        multiplier = self.dreambooth.world_size if params else 1  # check if tuning
        images = random.choices(
            self.controls(), k=self.dreambooth.params.images * multiplier
        )
        prompts = random.sample(
            self.dreambooth.params.prompts, k=self.dreambooth.params.images * multiplier
        )
        generation = GenerationRequest(
            images=images, prompts=prompts, params=params or {}
        )
        logger.info("Generation request: {}", generation)

        logger.info("Sending generation requests...")
        for i, queue in enumerate(self.dreambooth.queues.proc):
            logger.debug(f"Sending request to {i}...")
            queue.put_nowait(
                ProcessRequest(demographics=self.demographics, generation=generation)
            )

        logger.info("Receiving generation responses...")
        remaining = set(range(self.dreambooth.world_size))
        while remaining:
            logger.debug("Receiving response...")
            try:
                resp = self.dreambooth.queues.response.get(timeout=120)
                if isinstance(resp, ProcessResponseSentinel):
                    logger.info("Rank {} is done!", resp.rank)
                    remaining.remove(resp.rank)
                else:
                    logger.info("Received response: {}", resp)
                    yield resp
            except Empty:
                self.dreambooth.ctx.join(0)
                raise

    @torch.inference_mode()
    def generate(self):
        imgs = itertools.chain.from_iterable(r.images for r in self._generate())
        yield grid(self.controls() + list(imgs))

    @torch.inference_mode()
    def tune(self, params: dict[str, list[int | float]]):
        logger.info("Tuning with params: {}", params)
        grids = [
            grid(self.images()[: self.dreambooth.params.images]),
            grid(self.controls()[: self.dreambooth.params.images]),
        ]
        for resp in self._generate(params):
            img = grid(resp.images)
            text_kwargs = {
                "text": json.dumps(
                    {
                        "_".join(s[:3] for s in k.split("_")): round(v, 2)
                        for k, v in resp.params.items()
                    },
                ),
                "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                "fontScale": 1,
                "thickness": 2,
            }
            (_, text_w), _ = cv2.getTextSize(**text_kwargs)
            image = cv2.putText(
                img=np.array(img),
                org=(img.height // 10, img.width // 2 - text_w // 4),
                color=(255, 255, 255),
                lineType=cv2.LINE_AA,
                **text_kwargs,
            )
            img = Image.fromarray(image)
            grids.append(img)

        logger.info("Tuning complete!")
        yield grid(grids, w=8)
