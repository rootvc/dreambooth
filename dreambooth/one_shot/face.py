from collections import Counter
from functools import cache as f_cache
from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import keras
import numpy as np
import tensorflow as tf
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000 as cdiff
from colormath.color_objects import LabColor
from colormath.color_objects import sRGBColor as Color
from deepface import DeepFace
from deepface.detectors import OpenCvWrapper
from dreambooth_old.train.shared import grid
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field
from retinaface import RetinaFace
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants, tag_constants
from torchvision.transforms.functional import to_pil_image

from one_shot.params import Params, Settings
from one_shot.utils import collect

OpenCvWrapper.get_opencv_path = lambda: "/usr/local/share/opencv4/haarcascades/"


class Bounds(BaseModel):
    dims: tuple[int, ...]
    shape: tuple[int, ...]

    origin_x: int = Field(int, alias="x")
    origin_y: int = Field(int, alias="y")
    width: int = Field(int, alias="w")
    height: int = Field(int, alias="h")

    @property
    def mid_y(self):
        return self.shape[0] // 2

    @property
    def mid_x(self):
        return self.shape[1] // 2

    @property
    def quadrant(self):
        if self.origin_y < self.mid_y:
            if self.origin_x < self.mid_x:
                return 0
            else:
                return 1
        else:
            if self.origin_x < self.mid_x:
                return 2
            else:
                return 3

    def _slice(self, mask_padding: float):
        buffer_y = int(self.dims[0] * mask_padding)
        buffer_x = int(self.dims[1] * mask_padding)

        start_y = max(0, self.origin_y - buffer_y)
        end_y = min(self.shape[0], self.origin_y + self.height + buffer_y)
        start_x = max(0, self.origin_x - buffer_x)
        end_x = min(self.shape[1], self.origin_x + self.width + buffer_x)

        return (slice(start_y, end_y), slice(start_x, end_x))

    def slice(self, mask_padding: float):
        original_slices = self._slice(mask_padding)
        quadrant = self.quadrant

        if quadrant == 0:
            return original_slices
        elif quadrant == 1:
            return (original_slices[0], slice(0, original_slices[1].stop - self.mid_x))
        elif quadrant == 2:
            return (slice(0, original_slices[0].stop - self.mid_y), original_slices[1])
        elif quadrant == 3:
            return (
                slice(0, original_slices[0].stop - self.mid_y),
                slice(0, original_slices[1].stop - self.mid_x),
            )
        else:
            raise ValueError(f"Invalid quadrant: {quadrant}")


class CompiledModel:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.model = tf.saved_model.load(path, tags=[tag_constants.SERVING])
        self.graph = self.model.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.path})"

    @property
    def input_name(self):
        return self.graph.inputs[0].name.removesuffix(":0")

    def predict(self, inputs: Any, **_kwargs):
        return self.graph(**{self.input_name: inputs})["output_0"].numpy()


class Colors:
    HEX: dict[str, str] = {
        "52220B": "brown",
        "3C0C0A": "brown",
        "422020": "black",
        "803405": "light brown",
        "A46524": "amber",
        "8A9090": "gray",
        "6A984D": "green",
        "869078": "green",
        "728699": "blue",
        "83AED2": "light blue",
    }

    @staticmethod
    @f_cache
    def convert(hex: str) -> LabColor:
        return convert_color(Color.new_from_rgb_hex(hex), LabColor)

    @classmethod
    def closest(cls, color: Color):
        return min(
            cls.HEX.items(),
            key=lambda kv: cdiff(convert_color(color, LabColor), cls.convert(kv[0])),
        )[1]


class Face:
    MODELS = ("Gender", "Race")

    settings = Settings()
    cache: Path = Path(settings.cache_dir) / "face"

    def __init__(self, params: Params, images: list[Image.Image]):
        self.params = params
        self.images = images

    @cached_property
    def image(self):
        return np.asarray(grid(self.images))

    @f_cache
    def analyze(self, **kwargs):
        return DeepFace.analyze(
            self.image,
            actions=("gender", "race"),
            detector_backend="retinaface",
            **kwargs,
        )

    @property
    def dims(self):
        return np.asarray(self.images[0]).shape

    @f_cache
    @collect
    def faces(self):
        for face in self.analyze():
            bounds = Bounds(**face["region"], dims=self.dims, shape=self.image.shape)
            y, x = bounds.slice(self.params.mask_padding)
            image = np.asarray(self.images[bounds.quadrant])
            masked = np.zeros(self.dims, dtype=image.dtype)
            masked[y, x] = image[y, x]
            yield to_pil_image(masked)

    @f_cache
    def demographics(self):
        logger.info("Analyzing demographics...")
        try:
            faces = self.analyze()
        except Exception as e:
            logger.exception(e)
            return {"race": "beautiful", "gender": "person"}
        return {
            k.removeprefix("dominant_"): (
                Counter(r[k] for r in faces).most_common(1)[0][0]
            ).lower()
            for k in {"dominant_gender", "dominant_race"}
        }

    @collect
    def eye_masks(self):
        for face in self.images:
            img = np.asarray(face)
            logger.info("Detecting faces...")

            detected = RetinaFace.detect_faces(img)
            if not isinstance(detected, dict):
                yield face, None
                continue

            mask = np.zeros(img.shape, img.dtype)
            colors = []

            for face in detected.values():
                logger.info("Face: {}", face)

                for eye in (
                    face["landmarks"]["right_eye"],
                    face["landmarks"]["left_eye"],
                ):
                    bounds = Bounds(
                        dims=self.dims, shape=self.dims, x=eye[0], y=eye[1], h=0, w=0
                    )
                    y, x = bounds._slice(self.params.mask_padding)
                    mask[y, x] = 255
                    colors.append(
                        Color(*np.mean(img[y, x], axis=(0, 1)), is_upscaled=True)
                    )

            yield to_pil_image(mask), Counter(
                Colors.closest(c) for c in colors
            ).most_common(1)[0][0]

    @classmethod
    def load_models(cls):
        if not hasattr(DeepFace, "model_obj"):
            DeepFace.model_obj = {}
        for name in cls.MODELS:
            try:
                logger.info("Loading {}...", name)
                DeepFace.model_obj[name] = CompiledModel(cls.cache / name)
            except Exception as e:
                logger.exception(e)
                logger.warning("Failed to load {}", name)

    def compile_models(self):
        self.analyze(enforce_detection=False, align=False)
        content = self.faces()[0]

        def input_fn() -> Generator[tuple[np.ndarray], None, None]:
            yield (np.asarray(content),)

        for name in self.MODELS:
            if (self.cache / name).exists():
                logger.info(f"Skipping {name}...")
                continue
            logger.info(f"Compiling {name}...")
            model: keras.Model = DeepFace.model_obj[name]
            with TemporaryDirectory() as dir:
                model.export(dir)
                converter = trt.TrtGraphConverterV2(
                    input_saved_model_dir=dir, precision_mode=trt.TrtPrecisionMode.FP16
                )
                logger.info("Converting...")
                converter.convert()
                logger.info("Building...")
                converter.build(input_fn=input_fn)
                converter.save(self.cache / name)
                converter.summary()
