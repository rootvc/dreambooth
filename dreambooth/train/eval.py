import hashlib
import itertools
import shutil
from datetime import timedelta
from functools import cached_property
from typing import Optional, TypeVar, cast

import cv2
import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import wandb
from basicsr.utils import img2tensor, tensor2img
from diffusers import StableDiffusionLatentUpscalePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionDepth2ImgPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_depth2img import (
    preprocess,
)
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TT
from torchvision.transforms.functional import normalize, pil_to_tensor, to_pil_image
from torchvision.utils import make_grid

from dreambooth.params import Class, HyperParams
from dreambooth.registry import CompiledModelsRegistry
from dreambooth.train.shared import (
    convert_image,
    depth_image_path,
    depth_transforms,
    dprint,
    image_transforms,
    is_main,
    main_process_only,
    patch_allowed_pipeline_classes,
)
from dreambooth.train.shared import (
    images as get_images,
)
from dreambooth.vendor.codeformer.codeformer_arch import CodeFormer

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
        pipeline: StableDiffusionDepth2ImgPipeline,
    ):
        self.params = params
        self.instance_class = instance_class
        self.device = device
        self.pipeline = pipeline

    @cached_property
    def test_images(self):
        transforms = TT.Compose(
            [
                image_transforms(
                    self.params.model.resolution,
                    augment=False,
                    to_pil=True,
                    normalize=True,
                ),
                TT.Grayscale(num_output_channels=3),
                TT.GaussianBlur(5, sigma=2.0),
            ]
        )
        return [transforms(Image.open(p)) for p in get_images(self.instance_class.data)]

    @cached_property
    def depth_images(self):
        transforms = depth_transforms(
            self.params.model.resolution, self.pipeline.vae_scale_factor
        )
        return [
            transforms(Image.open(depth_image_path(p))).to(
                self.device, dtype=self.params.dtype
            )
            for p in get_images(self.instance_class.data)
        ]

    def _upsampler(self) -> StableDiffusionLatentUpscalePipeline:
        with patch_allowed_pipeline_classes():
            upsampler = StableDiffusionLatentUpscalePipeline.from_pretrained(
                self.params.upscale_model,
                torch_dtype=self.params.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
        upsampler.enable_xformers_memory_efficient_attention()
        return upsampler

    def _restorer(self):
        model = CompiledModelsRegistry.get(CodeFormer).to(self.device)
        model.load_state_dict(
            torch.load("weights/CodeFormer/codeformer.pth")["params_ema"]
        )
        return model.eval()

    def _face_helper(self) -> FaceRestoreHelper:
        helper = FaceRestoreHelper(
            self.params.upscale_factor,
            det_model="retinaface_resnet50",
            use_parse=True,
            device=self.device,
        )
        helper.clean_all()
        return helper

    def _convert_image(self, pil_image: Image.Image) -> np.ndarray:
        return convert_image(pil_image)

    def _extract_face(
        self, helper: FaceRestoreHelper, image: np.ndarray
    ) -> torch.Tensor:
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
        return face_t.unsqueeze(0).to(self.device)

    def _restore_face(
        self,
        restorer: torch.nn.Module,
        helper: FaceRestoreHelper,
        cropped_t: torch.Tensor,
    ):
        output = restorer(cropped_t, w=self.params.fidelity_weight, adain=True)[0]
        restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        helper.add_restored_face(restored.astype("uint8"))

        del output
        torch.cuda.empty_cache()

    def _paste_face(
        self, helper: FaceRestoreHelper, pil_image: Image.Image, image: np.ndarray
    ) -> np.ndarray:
        dprint("Enhancing face...")
        helper.get_inverse_affine()
        return helper.paste_faces_to_input_image(upsample_img=image)

    def _gen_images(self) -> list[tuple[str, Image.Image]]:
        loader = DataLoader(
            PromptDataset(self.params),
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
            batch_size=len(self.params.eval_prompts),
        )
        image = preprocess(self.test_images[0]).to(self.device, dtype=self.params.dtype)

        all_images = []
        for batch in loader:
            images = self.pipeline(
                batch,
                image=[image] * len(batch),
                depth_map=self.depth_images[0],
                negative_prompt=[self.params.negative_prompt] * len(batch),
                num_inference_steps=self.params.test_steps,
                guidance_scale=self.params.test_guidance_scale,
                strength=self.params.test_strength,
            ).images
            all_images.extend(zip(batch, images))

        return all_images

    @torch.inference_mode()
    def _process_image(self, pil_image: Image.Image) -> Optional[np.ndarray]:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            restorer = self._restorer()

            helper = self._face_helper()
            dprint("Converting image...")
            image = self._convert_image(pil_image)
            dprint("Extracting face...")
            try:
                face = self._extract_face(helper, image)
            except ValueError:
                return None
            dprint("Restoring face...")
            self._restore_face(restorer, helper, face)
            dprint("Pasting face...")
            return self._paste_face(helper, pil_image, image)

    @main_process_only
    def _upload_images(self):
        if self.params.debug_outputs:
            wandb.run.log(
                {
                    "source": [
                        wandb.Image(self.test_images[0], caption="image"),
                        wandb.Image(self.depth_images[0], caption="depth"),
                    ]
                }
            )
        if self.params.debug_outputs:
            keys = ("original", "restored")
        elif self.params.restore_faces:
            keys = ("restored",)
        else:
            keys = ("original",)
        for key in keys:
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

    def _grid(self, images: list[Image.Image]) -> Image.Image:
        tensors = torch.stack([pil_to_tensor(img) for img in images])
        grid = make_grid(tensors, nrow=2)
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

    @torch.inference_mode()
    def generate(self):
        dprint("Generating images...")
        prompts, images = cast(
            tuple[tuple[str], tuple[Image.Image]], list(zip(*self._gen_images()))
        )
        prompt_path, original_path, restored_path = self._paths()
        self.wait_for_everyone()

        if self.params.upscale_factor > 1:
            dprint("Upscaling images...")
            upsampler = self._upsampler()
            size: int = self.params.model.resolution * self.params.upscale_factor
            prompt = [
                p.replace(self.params.token, self.params.source_token) for p in prompts
            ]
            image = [i.resize((size, size), resample=Image.LANCZOS) for i in images]
            images = upsampler(prompt=prompt, image=image).images
            del upsampler

        dprint("Cleaning up...")
        del self.pipeline

        if self.params.restore_faces:
            dprint("Restoring images...")
            with multiprocessing.Pool(len(self.params.eval_prompts)) as p:
                all_restored = iter(p.map(self._process_image, images))
        else:
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
