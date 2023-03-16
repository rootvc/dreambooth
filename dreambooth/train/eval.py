import hashlib
import itertools
import os
import shutil
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar

import cv2
import numpy as np
import torch
import torch.distributed
import wandb
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator
from dreambooth.train.shared import (
    compile_model,
    dprint,
    main_process_only,
)

T = TypeVar("T", bound=torch.nn.Module)


class PromptDataset(Dataset):
    def __init__(self, params: HyperParams):
        self.params = params

    def __len__(self):
        return len(self.params.eval_prompts)

    def __getitem__(self, i: int):
        return [self.params.eval_prompts[i]]


class Evaluator:
    def __init__(
        self,
        accelerator: BaseAccelerator,
        params: HyperParams,
        pipeline: StableDiffusionPipeline,
    ):
        self.params = params
        self.accelerator = accelerator
        self.pipeline = pipeline

    def compile(self, model: torch.nn.Module, **kwargs):
        return compile_model(model, backend=self.params.dynamo_backend, **kwargs)

    def _upsampler(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=self.params.upscale_factor).to(
            self.accelerator.device
        )
        os.system(f"ls -la {Path('./weights').absolute()}")
        os.system(f"ls -la {Path('./weights/.').absolute()}")
        os.system(f"ls -lL {Path('./weights/').absolute()}")
        os.system(f"ls -lH {Path('./weights/').absolute()}")
        os.system(f"ls -la {Path('./weights/realesrgan').absolute()}")
        upsampler = RealESRGANer(
            scale=self.params.upscale_factor,
            model_path=str(
                Path("./weights/realesrgan/RealESRGAN_x2plus.pth").absolute()
            ),
            model=model.eval(),
            pre_pad=0,
            half=True,
            device=self.accelerator.device,
        )
        upsampler.model = self.compile(upsampler.model)
        return upsampler

    def _restorer(self):
        model = ARCH_REGISTRY.get("CodeFormer")().to(self.accelerator.device)
        model.load_state_dict(
            torch.load("weights/CodeFormer/codeformer.pth")["params_ema"]
        )
        return self.compile(model.eval())

    def _face_helper_singleton(self) -> FaceRestoreHelper:
        if not hasattr(self, "__face_helper"):
            self.__face_helper = FaceRestoreHelper(
                self.params.upscale_factor,
                det_model="dlib",
                use_parse=True,
                device=self.accelerator.device,
            )
        return self.__face_helper

    def _face_helper(self) -> FaceRestoreHelper:
        helper = self._face_helper_singleton()
        helper.clean_all()
        return helper

    def _convert_image(self, pil_image: Image) -> np.ndarray:
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _extract_face(
        self, helper: FaceRestoreHelper, image: np.ndarray
    ) -> tuple[np.ndarray, torch.Tensor]:
        helper.read_image(image)
        if (
            helper.get_face_landmarks_5(
                only_keep_largest=True,
                resize=int(self.params.model.resolution * 1.25),
            )
            != 1
        ):
            raise ValueError("No face detected")
        helper.align_warp_face()
        face = helper.cropped_faces[0]

        face_t: torch.Tensor = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
        normalize(face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        return (face, face_t.unsqueeze(0).to(self.accelerator.device))

    def _restore_face(
        self,
        restorer: torch.nn.Module,
        helper: FaceRestoreHelper,
        cropped: np.ndarray,
        cropped_t: torch.Tensor,
    ):
        output = restorer(cropped_t, w=self.params.fidelity_weight, adain=True)[0]
        restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        helper.add_restored_face(restored, cropped)

        del output
        torch.cuda.empty_cache()

    def _paste_face(
        self,
        upsampler: RealESRGANer,
        helper: FaceRestoreHelper,
        image: np.ndarray,
    ) -> np.ndarray:
        background = upsampler.enhance(image, outscale=self.params.upscale_factor)[0]
        helper.get_inverse_affine()
        return helper.paste_faces_to_input_image(
            upsample_img=background, face_upsampler=upsampler
        )

    def _gen_images(self) -> list[tuple[str, Image]]:
        loader = DataLoader(
            PromptDataset(self.params),
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
            batch_size=self.params.batch_size,
        )
        loader = self.accelerator.prepare(loader)

        all_images = []
        for batch in loader:
            images = self.pipeline(
                batch,
                negative_prompt=[self.params.negative_prompt] * len(batch),
                num_inference_steps=self.params.test_steps,
                guidance_scale=self.params.test_guidance_scale,
            ).images
            all_images.extend(zip(batch, images))

        return all_images

    def _process_image(
        self,
        restorer: torch.nn.Module,
        upsampler: RealESRGANer,
        pil_image: Image,
    ) -> Optional[np.ndarray]:
        helper = self._face_helper()
        dprint("Converting image...")
        image = self._convert_image(pil_image)
        dprint("Extracting face...")
        try:
            face, face_t = self._extract_face(helper, image)
        except ValueError:
            return None
        dprint("Restoring face...")
        self._restore_face(restorer, helper, face, face_t)
        dprint("Pasting face...")
        return self._paste_face(upsampler, helper, image)

    @main_process_only
    def _upload_images(self):
        for key in ("original", "restored"):
            self.accelerator.wandb_tracker.log(
                {
                    key: [
                        wandb.Image(
                            str(p),
                            caption=(self.params.image_output_path / "prompt" / p.stem)
                            .with_suffix(".txt")
                            .read_text(),
                        )
                        for p in (self.params.image_output_path / key).glob("*.png")
                    ]
                }
            )

    def _paths(self):
        paths = [
            self.params.image_output_path / "prompt",
            self.params.image_output_path / "original",
            self.params.image_output_path / "restored",
        ]
        if self.accelerator.is_main_process:
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(exist_ok=True)
        return paths

    @torch.inference_mode()
    def generate(self):
        dprint("Generating images...")
        images = self._gen_images()
        del self.pipeline
        self.accelerator.free_memory()

        prompt_path, original_path, restored_path = self._paths()
        self.accelerator.wait_for_everyone()

        dprint("Compiling face models...")
        upsampler, restorer = self._upsampler(), self._restorer()
        restore = partial(self._process_image, restorer, upsampler)

        dprint(f"Saving {len(images)} images...")
        for prompt, image in images:
            dprint(prompt)
            slug = hashlib.md5(prompt.encode()).hexdigest()
            (prompt_path / f"{slug}.txt").write_text(prompt)
            image.save(original_path / f"{slug}.png")

            dprint("Restoring...")
            if (restored := restore(image)) is None:
                continue
            path = str(restored_path / f"{slug}.png")
            cv2.imwrite(path, restored)

        dprint("Waiting for upload...")
        torch.distributed.barrier(async_op=True).wait(timeout=timedelta(seconds=30))
        self._upload_images()
        dprint("Done!")
