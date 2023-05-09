import hashlib
import itertools
import shutil
from datetime import timedelta
from functools import cached_property
from typing import TypeVar, cast

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import tqdm
import wandb
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import pil_to_tensor, resize, to_pil_image
from torchvision.utils import make_grid

from dreambooth.params import Class, HyperParams
from dreambooth.train.shared import (
    dprint,
    is_main,
    main_process_only,
)
from dreambooth.train.shared import (
    images as get_images,
)

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

multiprocessing = torch.multiprocessing.get_context("forkserver")

T = TypeVar("T", bound=torch.nn.Module)


class PromptDataset(Dataset):
    def __init__(self, params: HyperParams):
        self.params = params

    def __iter__(self):
        return iter(self.params.eval_prompts)

    def __len__(self):
        return len(self.params.eval_prompts)

    def __getitem__(self, i: int):
        return [self.params.eval_prompts[i]]


class Evaluator:
    def __init__(
        self,
        device: torch.device,
        params: HyperParams,
        instance_class: Class,
        pipeline: StableDiffusionControlNetPipeline,
    ):
        self.params = params
        self.instance_class = instance_class
        self.device = device
        self.pipeline = pipeline

    def face_detector(self) -> FaceDetector:
        options = FaceDetectorOptions(
            base_options=BaseOptions(
                model_asset_path="weights/mediapipe/blaze_face_short_range.tflite"
            ),
            running_mode=VisionRunningMode.IMAGE,
        )
        return FaceDetector.create_from_options(options)

    def _face_bounding_box(self, image: np.ndarray):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self.face_detector().detect(mp_image)
        box = result.detections[0].bounding_box

        buffer_y = int(image.shape[0] * self.params.mask_padding)
        buffer_x = int(image.shape[1] * self.params.mask_padding)

        start_y = max(0, box.origin_y - buffer_y)
        end_y = min(image.shape[0], box.origin_y + box.height + buffer_y)
        start_x = max(0, box.origin_x - buffer_x)
        end_x = min(image.shape[1], box.origin_x + box.width + buffer_x)

        return (slice(start_y, end_y), slice(start_x, end_x))

    def _mask(self, src: np.ndarray, dest: np.ndarray):
        y, x = self._face_bounding_box(src)
        masked = np.zeros(dest.shape, dest.dtype)
        masked[y, x] = dest[y, x]
        return masked

    def _canny(self, image: np.ndarray, sigma=0.5):
        med = np.median(image)
        lower = int(max(0.0, (1.0 - sigma) * med))
        upper = int(min(255.0, (1.0 + sigma) * med))
        canny = cv2.Canny(image, lower, upper)

        canny = canny[:, :, None]
        return np.concatenate([canny, canny, canny], axis=2)

    def _preprocess(self, pil_image: Image.Image):
        resized = resize(
            pil_image, (self.params.model.resolution, self.params.model.resolution)
        )
        image = np.asarray(resized)
        canny = self._canny(image)
        masked = self._mask(image, canny)
        return Image.fromarray(masked)

    @cached_property
    def test_images(self):
        return [Image.open(p) for p in get_images(self.instance_class.data)]

    @cached_property
    def cond_images(self):
        dprint("Preprocessing images...")
        images = [
            self._preprocess(Image.open(p))
            for p in get_images(self.instance_class.data)
        ]
        print("Done preprocessing...")
        return images

    def _gen_images(self) -> list[tuple[str, Image.Image]]:
        loader = DataLoader(
            PromptDataset(self.params),
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
            batch_size=4,  # len(self.params.eval_prompts),
        )
        # preprocess(self.test_images[0]).to(self.device, dtype=self.params.dtype)

        all_images = []
        for batch in tqdm.tqdm(loader):
            images = self.pipeline(
                batch,
                width=self.params.model.resolution,
                height=self.params.model.resolution,
                image=[self.cond_images[0]] * len(batch),
                negative_prompt=[self.params.negative_prompt] * len(batch),
                num_inference_steps=self.params.test_steps,
                guidance_scale=self.params.test_guidance_scale,
                controlnet_conditioning_scale=self.params.test_strength,
            ).images
            all_images.extend(zip(batch, images))

        return all_images

    @main_process_only
    def _upload_images(self):
        if self.params.debug_outputs:
            wandb.run.log(
                {
                    "source": [
                        wandb.Image(self.test_images[0], caption="image"),
                        wandb.Image(self.cond_images[0], caption="condition"),
                    ]
                }
            )
        wandb.run.log(
            {
                "original": [
                    wandb.Image(
                        str(p),
                        caption=(self.params.image_output_path / "prompt" / p.stem)
                        .with_suffix(".txt")
                        .read_text(),
                    )
                    for p in (self.params.image_output_path / "original").glob("*.png")
                ]
            }
        )

    def _grid(self, images: list[Image.Image]) -> Image.Image:
        tensors = torch.stack([pil_to_tensor(img) for img in images])
        grid = make_grid(tensors, nrow=2, pad_value=255, padding=10)
        return to_pil_image(grid)

    def _paths(self):
        paths = [
            self.params.image_output_path / "prompt",
            self.params.image_output_path / "original",
            self.params.image_output_path / "restored",
        ]
        if is_main():
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(exist_ok=True)
        return paths

    def wait_for_everyone(self):
        if torch.distributed.is_initialized():
            dprint("Waiting for other processes to finish...")
            torch.distributed.barrier(async_op=True).wait(timeout=timedelta(seconds=30))

    @torch.no_grad()
    def generate(self):
        dprint("Generating images...")
        prompts, images = cast(
            tuple[tuple[str], tuple[Image.Image]], list(zip(*self._gen_images()))
        )
        prompt_path, original_path, restored_path = self._paths()
        self.wait_for_everyone()

        dprint("Cleaning up...")
        del self.pipeline

        all_restored = itertools.repeat(None)

        dprint(f"Saving {len(images)} images...")
        for prompt, image in zip(prompts, images):
            dprint(prompt)
            slug = hashlib.md5(prompt.encode()).hexdigest()
            (prompt_path / f"{slug}.txt").write_text(prompt)
            image.save(original_path / f"{slug}.png")

            dprint("Restoring...")
            if (restored := next(all_restored)) is None:
                continue
            path = str(restored_path / f"{slug}.png")
            cv2.imwrite(path, restored)

        dprint("Saving grid...")
        grid = self._grid(images)
        grid.save(self.params.image_output_path / "grid.png")

        dprint("Waiting for upload...")
        self.wait_for_everyone()
        self._upload_images()
        dprint("Done!")
