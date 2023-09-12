from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Generator

import keras
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from dreambooth_old.train.shared import grid
from loguru import logger
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants, tag_constants

from one_shot.params import Params, Settings


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


class Face:
    MODELS = ("Gender", "Race")

    settings = Settings()
    cache: Path = Path(settings.cache_dir) / "face"

    def __init__(self, params: Params, images: list[np.ndarray]):
        self.params = params
        self.images = images

    def _demographics(self, **kwargs):
        return DeepFace.analyze(
            np.array(grid(self.images)),
            actions=("gender", "race"),
            detector_backend="mediapipe",
        )

    def demographics(self):
        logger.info("Analyzing demographics...")
        try:
            res = self._demographics()
        except Exception as e:
            logger.exception(e)
            return {"race": "beautiful", "gender": "person"}
        logger.info("Demographics: {}", res)
        return {
            k.removeprefix("dominant_"): Counter(r[k] for r in res).most_common(1)[0][0]
            for k in {"dominant_gender", "dominant_race"}
        }

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
        self._demographics(enforce_detection=False, align=False)
        content, _, _ = DeepFace.functions.extract_faces(
            self.images[0], detector_backend="skip"
        )[0]

        def input_fn() -> Generator[tuple[np.ndarray], None, None]:
            yield (content,)

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
