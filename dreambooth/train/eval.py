import hashlib
import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor as Pool
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Optional, TypeVar

import cv2
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import wandb
from accelerate.utils import convert_outputs_to_fp32
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
from dreambooth.train.shared import (
    compile_model,
    dprint,
    is_main,
    main_process_only,
)

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
        pipeline: StableDiffusionPipeline,
    ):
        self.params = params
        self.device = device
        self.pipeline = pipeline

    def compile(self, model: torch.nn.Module, **kwargs):
        return compile_model(model, backend=self.params.dynamo_backend, **kwargs)

    def _upsampler(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=self.params.upscale_factor).to(
            self.device, dtype=self.params.dtype
        )
        upsampler = RealESRGANer(
            scale=self.params.upscale_factor,
            model_path=str(
                Path("./weights/realesrgan/RealESRGAN_x2plus.pth").absolute()
            ),
            model=model.eval(),
            pre_pad=0,
            device=self.device,
            half=True,
        )
        return upsampler

    def _restorer(self):
        model = ARCH_REGISTRY.get("CodeFormer")().to(self.device)
        model.load_state_dict(
            torch.load("weights/CodeFormer/codeformer.pth")["params_ema"]
        )
        return model.eval()

    def _face_helper_singleton(self) -> FaceRestoreHelper:
        if not hasattr(self, "__face_helper"):
            self.__face_helper = FaceRestoreHelper(
                self.params.upscale_factor,
                det_model="dlib",
                use_parse=True,
                device=self.device,
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
        return (face, face_t.unsqueeze(0).to(self.device))

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
        dprint("Enhancing background...")
        background = upsampler.enhance(image, outscale=self.params.upscale_factor)[0]
        helper.get_inverse_affine()
        dprint("Enhancing face...")
        return helper.paste_faces_to_input_image(
            upsample_img=background, face_upsampler=upsampler
        )

    def _gen_images(self) -> list[tuple[str, Image]]:
        loader = DataLoader(
            PromptDataset(self.params),
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
            batch_size=len(self.params.eval_prompts),
        )

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
        upsampler.model.forward = torch.cuda.amp.autocast()(upsampler.model.forward)
        upsampler.model.forward = convert_outputs_to_fp32(upsampler.model.forward)
        upsampler.model = self.compile(upsampler.model).to(self.device)

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
            wandb.run.log(
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
        if is_main():
            for path in paths:
                shutil.rmtree(path, ignore_errors=True)
                path.mkdir(exist_ok=True)
        return paths

    def wait_for_everyone(self):
        if torch.distributed.is_initialized():
            torch.distributed.barrier(async_op=True).wait(timeout=timedelta(seconds=30))

    @torch.inference_mode()
    def generate(self):
        dprint("Generating images...")
        images = self._gen_images()
        del self.pipeline
        torch.cuda.empty_cache()

        prompt_path, original_path, restored_path = self._paths()
        self.wait_for_everyone()

        dprint("Compiling face models...")
        upsampler, restorer = self._upsampler(), self._restorer()
        restore = partial(self._process_image, restorer, upsampler)

        with Pool(len(images), mp_context=multiprocessing) as p:
            all_restored = p.map(restore, (image for _, image in images))

        dprint(f"Saving {len(images)} images...")
        for i, (prompt, image) in enumerate(images):
            dprint(prompt)
            slug = hashlib.md5(prompt.encode()).hexdigest()
            (prompt_path / f"{slug}.txt").write_text(prompt)
            image.save(original_path / f"{slug}.png")

            dprint("Restoring...")
            if (restored := next(all_restored)) is None:
                continue
            path = str(restored_path / f"{slug}.png")
            cv2.imwrite(path, restored)

        dprint("Waiting for upload...")
        self.wait_for_everyone()
        self._upload_images()
        dprint("Done!")
