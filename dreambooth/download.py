import os
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Type
from urllib.parse import urlencode

import requests
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from cloudpathlib import CloudPath
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
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

from dreambooth.param.model import Model

HF_MODEL_CACHE = os.getenv("HF_MODEL_CACHE")
BUCKET = "s3://rootvc-photobooth"


def persist_model(model, name: str):
    if HF_MODEL_CACHE:
        path = Path(HF_MODEL_CACHE) / name
        print(f"Saving {name} to {path}")
        model.save_pretrained(path, safe_serialization=True)
        return model

    s3_path = CloudPath(BUCKET) / "models" / name
    if s3_path.exists():
        print(f"Model {name} already exists, skipping")
    else:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / name
            print(f"Saving {name} to {path}")
            model.save_pretrained(path, safe_serialization=True)
            s3_path.upload_from(path)
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


def download_hf_model(model: Model):
    models = [
        download(
            StableDiffusionXLControlNetPipeline,
            model.name,
            variant=model.variant,
            revision=model.revision,
            controlnet=download(
                ControlNetModel,
                model.control_net,
            ),
        )
    ]
    if model.vae:
        models.append(download(AutoencoderKL, model.vae))
    if model.refiner:
        models.append(
            download(
                StableDiffusionXLImg2ImgPipeline, model.refiner, variant=model.variant
            )
        )
    return models


def download_civitai_model(model: Model):
    files = {}
    with ExitStack() as stack:
        for query in [
            {"type": "Model", "format": "SafeTensor", "size": "full", "fp": "fp32"},
            {"type": "Config", "format": "Other"},
            {"type": "VAE", "format": "SafeTensor"},
        ]:
            f = stack.enter_context(NamedTemporaryFile())
            r = requests.get(
                f"https://civitai.com/api/download/models/{model.name}?{urlencode(query)}"
            )
            if r.status_code != 200:
                continue
            f.write(r.content)
            f.flush()
            files[query["type"]] = f.name

        if model.vae:
            vae = download(AutoencoderKL, model.vae)
        elif "VAE" in files:
            config = OmegaConf.load(files["Config"])
            vae_config = create_vae_diffusers_config(
                config, image_size=model.resolution
            )
            vae_config["scaling_factor"] = config.model.scale_factor

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
            image_size=model.resolution,
            prediction_type="epsilon",
            extract_ema=False,
            from_safetensors=True,
            vae=vae,
        )
        return [
            persist_model(pipe, model.name),
            download(
                ControlNetModel,
                model.control_net,
            ),
        ]


def download_model():
    model = Model()
    if model.source == "hf":
        return download_hf_model(model)
    else:
        return download_civitai_model(model)


if __name__ == "__main__":
    download_model()
    # download_test_models(None, HyperParams().test_model)
