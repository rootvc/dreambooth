import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Type
from urllib.parse import urlencode

import requests
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
    create_vae_diffusers_config,
    download_from_original_stable_diffusion_ckpt,
)
from omegaconf import OmegaConf
from safetensors.torch import load_file as safe_load
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from dreambooth.params import HyperParams

HF_MODEL_CACHE = os.getenv("HF_MODEL_CACHE")


def persist_model(model, name: str):
    if HF_MODEL_CACHE:
        path = Path(HF_MODEL_CACHE) / name
        print(f"Saving {name} to {path}")
        model.save_pretrained(path, safe_serialization=True)
    return model


def download(klass: Type, name: str, **kwargs):
    model = klass.from_pretrained(name, **kwargs)
    return persist_model(model, name)


def download_test_models(_, name: str):
    return [
        download(klass, name)
        for klass in [
            CLIPProcessor,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            CLIPVisionModelWithProjection,
            CLIPModel,
        ]
    ]


def download_hf_model(params: HyperParams):
    models = [
        download(
            StableDiffusionControlNetPipeline,
            params.model.name,
            variant=params.model.variant,
            revision=params.model.revision,
            torch_dtype=params.dtype,
            controlnet=download(
                ControlNetModel,
                params.model.control_net,
                torch_dtype=params.dtype,
            ),
        )
    ]
    if params.model.vae:
        models.append(download(AutoencoderKL, params.model.vae))
    return models


def download_civitai_model(params: HyperParams):
    files = {}
    with ExitStack() as stack:
        for query in [
            {"type": "Model", "format": "SafeTensor", "size": "full", "fp": "fp32"},
            {"type": "Config", "format": "Other"},
            {"type": "VAE", "format": "SafeTensor"},
        ]:
            f = stack.enter_context(NamedTemporaryFile())
            r = requests.get(
                f"https://civitai.com/api/download/models/{params.model.name}?{urlencode(query)}"
            )
            if r.status_code != 200:
                continue
            f.write(r.content)
            f.flush()
            files[query["type"]] = f.name

        if params.model.vae:
            vae = download(AutoencoderKL, params.model.vae)
        elif "VAE" in files:
            config = OmegaConf.load(files["Config"])
            vae_config = create_vae_diffusers_config(
                config, image_size=params.model.resolution
            )
            vae_config["scaling_factor"] = config.model.params.scale_factor

            checkpoint = safe_load(files["VAE"])
            for key in list(checkpoint.keys()):
                checkpoint[f"first_stage_model.{key}"] = checkpoint.pop(key)
            vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

            with init_empty_weights():
                vae = AutoencoderKL(**vae_config)
            for name, param in vae_checkpoint.items():
                set_module_tensor_to_device(vae, name, "cpu", value=param)
        else:
            vae = None

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path=files["Model"],
            original_config_file=files.get("Config"),
            image_size=params.model.resolution,
            prediction_type="epsilon",
            extract_ema=False,
            from_safetensors=True,
            vae=vae,
        )
        return [
            persist_model(pipe, params.model.name),
            download(
                ControlNetModel,
                params.model.control_net,
                torch_dtype=params.dtype,
            ),
        ]


def download_model():
    params = HyperParams()
    if params.model.source == "hf":
        return download_hf_model(params)
    else:
        return download_civitai_model(params)


if __name__ == "__main__":
    download_model()
    # download_test_models(None, HyperParams().test_model)
