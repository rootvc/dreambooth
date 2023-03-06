import itertools
import json
import re
from functools import partial
from pathlib import Path
from typing import Iterable, TypeVar

import cv2
import numpy as np
import torch
import wandb
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from peft import LoraConfig, LoraModel, set_peft_model_state_dict
from PIL.Image import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize
from transformers import AutoTokenizer, CLIPTextModel

from dreambooth.params import HyperParams
from dreambooth.train.accelerators.base import BaseAccelerator
from dreambooth.train.shared import (
    compile_model,
    main_process_only,
    partition,
    patch_allowed_pipeline_classes,
)

T = TypeVar("T", bound=torch.nn.Module)


class PromptDataset(Dataset):
    def __init__(self, params: HyperParams):
        self.params = params

    def __len__(self):
        return len(self.params.eval_prompts)

    def __getitem__(self, i: int):
        return [self.params.eval_template.format(prompt=self.params.eval_prompts[i])]


class Evaluator:
    def __init__(self, accelerator: BaseAccelerator, params: HyperParams):
        self.params = params
        self.accelerator = accelerator

    @main_process_only
    def _print(self, *args, **kwargs):
        print(*args, **kwargs)

    def _init_text(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.params.model.name,
            revision=self.params.model.revision,
            subfolder="tokenizer",
            use_fast=False,
        )
        assert tokenizer.add_tokens(self.params.token) == 1

        text_encoder = (
            CLIPTextModel.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                subfolder="text_encoder",
                torch_dtype=self.params.dtype,
            )
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        text_encoder.resize_token_embeddings(len(tokenizer))

        token_id = tokenizer.convert_tokens_to_ids(self.params.token)
        embeds: torch.Tensor = text_encoder.get_input_embeddings().weight.data
        token_embedding = torch.load(
            self.params.model_output_path / "token_embedding.pt",
            map_location=self.accelerator.device,
        )

        embeds[token_id] = token_embedding[self.params.token]

        return tokenizer, text_encoder

    def _unet(self):
        unet = (
            UNet2DConditionModel.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                subfolder="unet",
                torch_dtype=self.params.dtype,
            )
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        return unet

    def _vae(self):
        vae = (
            AutoencoderKL.from_pretrained(
                self.params.model.vae, torch_dtype=self.params.dtype
            )
            .to(self.accelerator.device)
            .requires_grad_(False)
            .eval()
        )
        vae.enable_slicing()
        return vae

    def _load_pipeline(self) -> StableDiffusionPipeline:
        tokenizer, text_encoder = self._init_text()
        unet, vae = self._unet(), self._vae()
        device = self.accelerator.device

        config = json.loads(
            (self.params.model_output_path / "lora_config.json").read_text()
        )
        state = torch.load(
            self.params.model_output_path / "lora_weights.pt",
            map_location=device,
        )
        unet_state, text_state = partition(state, lambda kv: "text_encoder_" in kv[0])

        unet = LoraModel(LoraConfig(**config["unet_peft"]), unet).to(
            device, dtype=self.params.dtype
        )
        set_peft_model_state_dict(unet, unet_state)

        text_encoder = LoraModel(LoraConfig(**config["text_peft"]), text_encoder).to(
            device, dtype=self.params.dtype
        )
        set_peft_model_state_dict(
            text_encoder,
            {k.removeprefix("text_encoder_"): v for k, v in text_state.items()},
        )

        with patch_allowed_pipeline_classes():
            pipeline = DiffusionPipeline.from_pretrained(
                self.params.model.name,
                revision=self.params.model.revision,
                safety_checker=None,
                low_cpu_mem_usage=True,
                local_files_only=True,
                unet=compile_model(unet),
                text_encoder=compile_model(text_encoder),
                vae=compile_model(vae),
                tokenizer=tokenizer,
                torch_dtype=self.params.dtype,
            )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config
        )

        return pipeline

    def _upsampler(self):
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.params.upscale_factor,
        ).to(self.accelerator.device)
        path = Path("./weights/realesrgan/RealESRGAN_x2plus.pth")
        print(path, path.absolute())
        return RealESRGANer(
            scale=self.params.upscale_factor,
            model_path=str(path.absolute()),
            model=compile_model(model.eval()),
            pre_pad=0,
            half=True,
            device=self.accelerator.device,
        )

    def _restorer(self):
        model = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.accelerator.device)
        model.load_state_dict(
            torch.load("weights/codeformer/codeformer.pth")["params_ema"]
        )
        return compile_model(model.eval())

    def _face_helper_singleton(self) -> FaceRestoreHelper:
        if not hasattr(self, "__face_helper"):
            self.__face_helper = FaceRestoreHelper(
                self.params.upscale_factor,
                face_size=self.params.model.resolution,
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
    ) -> torch.Tensor:
        helper.read_image(image)
        if helper.get_face_landmarks_5(only_keep_largest=True) != 1:
            raise ValueError("No face detected")
        helper.align_warp_face()
        face = helper.cropped_faces[0]

        face_t: torch.Tensor = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
        normalize(face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        return face_t.unsqueeze(0).to(self.accelerator.device)

    def _restore_face(
        self,
        restorer: torch.nn.Module,
        helper: FaceRestoreHelper,
        cropped: torch.Tensor,
    ):
        output = restorer(cropped, w=self.params.fidelity_weight, adain=True)[0]
        restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
        helper.add_restored_face(restored, cropped)

        del output
        torch.cuda.empty_cache()

    def _paste_face(
        self,
        upsampler: RealESRGANer,
        helper: FaceRestoreHelper,
        image: np.ndarray,
    ):
        background = upsampler.enhance(image, outscale=self.params.upscale_factor)[0]
        helper.get_inverse_affine()
        return helper.paste_faces_to_input_image(
            upsample_img=background, face_upsampler=upsampler
        )

    def _gen_images(self) -> Iterable[tuple[str, Image]]:
        pipeline = self._load_pipeline()
        loader = DataLoader(
            PromptDataset(self.params),
            batch_size=self.params.batch_size,
            collate_fn=lambda x: list(itertools.chain.from_iterable(x)),
        )
        loader = self.accelerator.prepare(loader)

        self._print(f"Generating images with {len(loader)} batches...")
        all_images = []
        for i, prompts in enumerate(loader):
            self._print(f"Batch {i * torch.cuda.device_count()}/{len(loader)}")
            images = pipeline(
                prompts,
                negative_prompt=[self.params.negative_prompt] * len(prompts),
                num_inference_steps=self.params.validation_steps,
            ).images
            all_images.extend(zip(prompts, images))

        return all_images

    def _process_image(
        self,
        restorer: torch.nn.Module,
        upsampler: RealESRGANer,
        pil_image: Image,
    ):
        helper = self._face_helper()
        image = self._convert_image(pil_image)
        face = self._extract_face(helper, image)
        self._restore_face(restorer, helper, face)
        return self._paste_face(upsampler, helper, image)

    @torch.inference_mode()
    def generate(self):
        self._print("Generating images...")
        images = self._gen_images()
        self.accelerator.wait_for_everyone()
        torch.cuda.empty_cache()

        self._print("Restoring faces...")
        upsampler, restorer = self._upsampler(), self._restorer()
        restore = partial(self._process_image, restorer, upsampler)

        log = []
        for prompt, image in images:
            restored = restore(image)
            slug = re.sub(r"[^\w]+", "_", re.sub(r"[\(\)]+", "", prompt))[:30]
            path = str(self.params.image_output_path / f"{slug}.png")
            cv2.imwrite(path, restored)
            log.append(wandb.Image(path, caption=prompt))

        self.accelerator.wandb_tracker.log({"output": log})
