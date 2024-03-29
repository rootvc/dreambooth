import itertools
import json
import random
from functools import cached_property
from operator import itemgetter
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Generator, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from one_shot import logger
from one_shot.dreambooth.process.process import (
    GenerationRequest,
    ProcessRequest,
    ProcessResponseSentinel,
)
from one_shot.face import FaceHelper
from one_shot.utils import (
    Face,
    chunks,
    collect,
    draw_masks,
    grid,
    images,
    open_image,
)

if TYPE_CHECKING:
    from one_shot.dreambooth.one_shot_dreambooth import OneShotDreambooth


class Request:
    def __init__(self, dreambooth: "OneShotDreambooth", id: str):
        self.dreambooth = dreambooth
        self.id = id

    @cached_property
    def image_dir(self):
        dir = Path(self.dreambooth.exit_stack.enter_context(TemporaryDirectory()))
        (self.dreambooth.settings.bucket / "dataset" / self.id).download_to(dir)
        return dir

    @cached_property
    @collect
    def images(self) -> Generator[Image.Image, None, None]:
        logger.info("Loading images...")
        for path in images(self.image_dir):
            logger.debug(f"Loading {path}...")
            yield open_image(self.dreambooth.params, path)

    def faces(self) -> list[Image.Image]:
        return self.face.primary_faces()

    @cached_property
    def face(self):
        face = FaceHelper(
            self.dreambooth.params.copy(update={"mask_padding": 0.0}),
            self.dreambooth.models.face,
            self.images,
            conservative=False,
            logger=logger.bind(tag="face"),
        )
        face.prefer_landmarks = self.demographics.use_ip_adapter(
            map(itemgetter(1), self._face.primary_face_bounds())
        )
        logger.info("Prefer landmarks: {}", face.prefer_landmarks)
        return face

    @cached_property
    def _face(self):
        return FaceHelper(
            self.dreambooth.params,
            self.dreambooth.models.face,
            self.images,
            logger=logger.bind(tag="demographics"),
        )

    @cached_property
    def demographics(self):
        demos = self._face.demographics()
        logger.info(demos)
        return demos

    def _generate(
        self, params: Optional[dict[str, list[int | float]]] = None, throw: bool = False
    ):
        multiplier = self.dreambooth.world_size if params else 1  # check if tuning
        sample: list[tuple[Image.Image, Face]] = random.sample(
            list(
                zip(
                    self.faces(),
                    map(itemgetter(1), self.face.primary_face_bounds()),
                )
            ),
            k=self.dreambooth.params.images * multiplier,
        )
        images, face_bounds = map(list, zip(*sample))
        prompts = random.sample(
            self.dreambooth.params.prompts, k=self.dreambooth.params.images * multiplier
        )
        generation = GenerationRequest(
            images=images, faces=face_bounds, prompts=prompts, params=params or {}
        )

        logger.info("Sending generation requests...")
        for i, queue in enumerate(self.dreambooth.queues.proc):
            logger.debug(f"Sending request to {i}...")
            queue.put_nowait(
                ProcessRequest(
                    demographics=self.demographics,
                    generation=generation,
                    tuning=bool(params),
                )
            )

        logger.info("Receiving generation responses...")
        remaining = set(range(self.dreambooth.world_size))
        while remaining:
            logger.debug("Receiving response...")
            try:
                resp = self.dreambooth.queues.response.get(
                    timeout=180 if params else 60
                )
                if isinstance(resp, ProcessResponseSentinel):
                    logger.info("Rank {} is done!", resp.rank)
                    remaining.remove(resp.rank)
                else:
                    logger.info("Received response: {}", resp.rank)
                    yield resp
            except Empty as e:
                try:
                    self.dreambooth.ctx.join(0)
                except Exception as ex:
                    if throw:
                        raise ex
                    else:
                        logger.exception("Error in subprocess, ending...")
                        break
                else:
                    if throw:
                        raise e
                    else:
                        logger.exception("Timeout waiting for response, ending...")
                        break

    @torch.inference_mode()
    def generate(self):
        imgs = itertools.chain.from_iterable(r.images for r in self._generate())
        return grid(list(imgs))

    @torch.inference_mode()
    def tune(self, params: dict[str, list[int | float]]):
        logger.info("Tuning with params: {}", params)

        masks = [
            draw_masks(img, [face for i, face in self.face.face_bounds() if i == idx])
            for idx, img in enumerate(self.images)
        ]
        grids = [
            grid(self.images[: self.dreambooth.params.images]),
            grid(masks[: self.dreambooth.params.images]),
            grid(self.face.primary_faces()[: self.dreambooth.params.images]),
            grid(
                [x[0] for x in self.face.eye_masks()][: self.dreambooth.params.images]
            ),
        ]

        for resp in self._generate(params):
            for images in chunks(resp.images, 1):
                img = grid(images)
                text_kwargs = {
                    "text": json.dumps(
                        {
                            "_".join(s[:3] for s in k.split("_")): round(v, 2)
                            if isinstance(v, float)
                            else str(v)[:5]
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

        logger.info("Persisting tuning...")
        path = Path(self.dreambooth.settings.cache_dir) / "tune" / f"{self.id}.webp"
        path.parent.mkdir(parents=True, exist_ok=True)
        final = grid(grids, w=self.dreambooth.world_size)
        try:
            final.save(path)
        except Exception:
            path = path.with_suffix(".png")
            final.reduce(4).save(path, optimize=True)

        self.dreambooth.volume.commit()

        logger.info("Tuning complete!\n{}", path.stat())
        yield path
