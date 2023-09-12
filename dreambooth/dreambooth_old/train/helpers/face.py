import logging
from typing import Optional, TypeVar

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
from PIL import Image
from torchvision.transforms.functional import resize

from dreambooth_old.params import HyperParams

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

multiprocessing = torch.multiprocessing.get_context("forkserver")

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=torch.nn.Module)


class FaceHelper:
    def __init__(
        self, params: HyperParams, image: Image.Image | np.ndarray | torch.Tensor
    ):
        self.params = params
        self.image = image if isinstance(image, np.ndarray) else np.asarray(image)

    @classmethod
    def face_detector(cls):
        if not hasattr(cls, "_face_detector"):
            options = FaceDetectorOptions(
                base_options=BaseOptions(
                    model_asset_path="cache/mediapipe/blaze_face_short_range.tflite"
                ),
                running_mode=VisionRunningMode.IMAGE,
            )
            cls._face_detector = FaceDetector.create_from_options(options)
        return cls._face_detector

    @classmethod
    def bounding_box(cls, image: np.ndarray, mask_padding: float):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = cls.face_detector().detect(mp_image)
        box = result.detections[0].bounding_box

        buffer_y = int(image.shape[0] * mask_padding)
        buffer_x = int(image.shape[1] * mask_padding)

        start_y = max(0, box.origin_y - buffer_y)
        end_y = min(image.shape[0], box.origin_y + box.height + buffer_y)
        start_x = max(0, box.origin_x - buffer_x)
        end_x = min(image.shape[1], box.origin_x + box.width + buffer_x)

        return (slice(start_y, end_y), slice(start_x, end_x))

    @classmethod
    def _mask(cls, src: np.ndarray, dest: np.ndarray, mask_padding: float):
        y, x = cls.bounding_box(src, mask_padding)
        masked = np.zeros(dest.shape, dest.dtype)
        masked[y, x] = dest[y, x]
        return masked

    def mask(self, dest: Optional[np.ndarray] = None):
        return self._mask(
            self.image,
            dest if dest is not None else self.image,
            self.params.mask_padding,
        )

    def canny(self, sigma=0.5):
        np.median(self.image)
        # lower = int(max(0.0, (1.0 - sigma) * med))
        # upper = int(min(255.0, (1.0 + sigma) * med))
        canny = cv2.Canny(self.image, 100, 200)

        canny = canny[:, :, None]
        return np.concatenate([canny, canny, canny], axis=2)

    @classmethod
    def preprocess(cls, params: HyperParams, pil_img: Image.Image) -> Image.Image:
        resized = resize(pil_img, [params.model.resolution, params.model.resolution])
        helper = cls(params, resized)
        canny = helper.canny()
        try:
            masked = helper.mask(canny)
        except IndexError:
            masked = canny
        return Image.fromarray(masked)
