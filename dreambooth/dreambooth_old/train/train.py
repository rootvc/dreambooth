import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import torch
import torch.distributed
import torch.version

from dreambooth_old.train.base import get_params
from dreambooth_old.train.sd.train import get_model

if TYPE_CHECKING:
    from sagemaker_training import environment

from dreambooth_old.params import Class, HyperParams
from dreambooth_old.train.shared import dprint, hash_bytes

IGNORE_MODS = ["_functorch", "fmha", "torchvision", "tempfile"]
IGNORE_RE = r"|".join([rf"(.*)\.{mod}\.(.*)" for mod in IGNORE_MODS])
warnings.filterwarnings("ignore", module=IGNORE_RE)
for klass in [ImportWarning, DeprecationWarning, ResourceWarning]:
    warnings.filterwarnings("ignore", category=klass)


class Params(TypedDict):
    instance_path: Path
    params: HyperParams


def _download(src: Path, dst: str | Path):
    subprocess.run(
        [
            "rclone",
            "copy",
            "--progress",
            "--ignore-checksum",
            "--size-only",
            "--fast-list",
            "--human-readable",
            "--stats-one-line",
            "--transfers=256",
            "--disable-http2",
            "--checkers=256",
            "--multi-thread-streams=32",
            "--s3-env-auth",
            "--s3-region=us-west-2",
            "--s3-use-accelerate-endpoint",
            "--s3-no-check-bucket",
            "--s3-no-head",
            src.as_uri().replace("s3://", "s3:/"),
            dst,
        ]
    )


def _unpack_model(env: "environment.Environment", name: str):
    model_data = Path(env.channel_input_dirs["model"])
    model_dir = Path("models") / name

    if not env.is_main:
        return model_dir

    os.makedirs(model_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call(
            [
                "pixz",
                "-d",
                model_data / Path(name).with_suffix(".tpxz"),
                f.name,
            ]
        )
        shutil.unpack_archive(f.name, model_dir, format="tar")

    print(f"Unpacked {name} to {model_dir}...")
    return model_dir


def _unpack_eval_models(model_dir: Path):
    params = get_params()
    models = model_dir / params.eval_model_path

    if not models.exists():
        raise ValueError(f"Model directory {models} does not exist!")

    if not Path("weights").is_symlink():
        os.symlink(models, "weights", target_is_directory=True)


def _setup_global_cache():
    import torch._inductor.config

    path = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) / "global_cache"
    path.mkdir(exist_ok=True, parents=True)
    torch._inductor.config.global_cache_dir = path


def _persist_global_cache():
    cache_dir = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])

    local_cache = cache_dir / "local_cache"
    global_cache = cache_dir / "global_cache"

    dprint("Persisting local cache...")
    if not local_cache.exists():
        dprint("No local cache found, skipping...")
        return

    dinfo = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    vinfo = torch.version.cuda

    dprint("Reading local cache...")
    local_cache_data = json.loads(local_cache.read_text())
    if global_cache.exists():
        global_cache_data = json.loads(global_cache.read_text())
    else:
        dprint("No global cache found, creating...")
        global_cache_data = {dinfo: {vinfo: {}}}
    global_cache_data[dinfo][vinfo] = local_cache_data

    dprint("Writing global cache...")
    global_cache.write_text(json.dumps(global_cache_data))

    dprint("Removing local cache...")
    local_cache.unlink()


def sagemaker_params(env: "environment.Environment") -> Params:
    train_data = Path(env.channel_input_dirs["train"])
    prior_data = Path(env.channel_input_dirs["prior"])
    cache_data = Path(env.channel_input_dirs["cache"])
    output_data = Path(env.channel_input_dirs["output"])
    params = get_params()

    params.model.name = _unpack_model(env, params.model.name)
    if params.model.vae:
        params.model.vae = _unpack_model(env, params.model.vae)
    if params.model.refiner:
        params.model.refiner = _unpack_model(env, params.model.refiner)
    params.test_model = _unpack_model(env, params.test_model)
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(env.model_dir)

    _setup_global_cache()
    if env.is_main:
        shutil.copytree(cache_data, os.environ["CACHE_DIR"], dirs_exist_ok=True)
        _unpack_eval_models(Path(env.channel_input_dirs["model"]))

    return {"instance_path": train_data, "params": params}


def sagemaker_cleanup(env: "environment.Environment"):
    # No need to persist models
    shutil.rmtree(env.model_dir, ignore_errors=True)
    os.makedirs(env.model_dir, exist_ok=True)

    dprint("Persisting global cache...")
    _persist_global_cache()
    dprint("Copying cache back to S3...")
    subprocess.run(
        [
            "rsync",
            "-ahSD",
            "--no-whole-file",
            "--no-compress",
            "--stats",
            "--inplace",
            f"{os.environ['CACHE_DIR']}/",
            env.channel_input_dirs["cache"],
        ]
    )


def cache_path():
    from cloudpathlib import CloudPath

    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    dinfo = torch.cuda.get_device_properties(torch.cuda.current_device()).name
    vinfo = torch.version.cuda
    key = f"{dinfo}-{vinfo}-{torch.__version__}".replace(" ", "-").lower()
    path = bucket / "cache" / key

    if not path.exists():
        (path / ".keep").touch(exist_ok=True)
    return path.as_uri().replace("s3://", "s3:/")


def standalone_params(is_main: bool) -> Params:
    from cloudpathlib import CloudPath

    id = os.environ["DREAMBOOTH_ID"]
    bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])

    base_dir = Path("/opt/ml")
    base_tmp_dir = Path("/tmp/ml") / id
    base_tmp_dir.mkdir(parents=True, exist_ok=True)

    train_data = base_tmp_dir / "train"
    output_data = base_tmp_dir / "output"
    model_data = base_tmp_dir / "model"
    prior_data = base_dir / "prior"

    params = get_params()
    params.prior_class = Class(prompt_=params.prior_prompt, data=prior_data)
    params.image_output_path = output_data
    params.model_output_path = Path(model_data)

    params.model_output_path.mkdir(parents=True, exist_ok=True)
    params.image_output_path.mkdir(parents=True, exist_ok=True)

    if os.getenv("WARM", "0") == "1":
        params.ti_train_epochs = 2
        params.lora_train_epochs = 2
        params.test_images = 1

        dprint("Downloading cache from S3...")
        _download(cache_path(), os.environ["CACHE_DIR"])

    if not is_main:
        return {"instance_path": train_data, "params": params}

    train_data_path = bucket / "dataset" / id
    priors_path = bucket / "priors" / hash_bytes(HyperParams().prior_prompt.encode())

    dprint("Downloading priors and training data from S3...")
    for src, dst in {train_data_path: train_data, priors_path: prior_data}.items():
        if src.exists() and not dst.exists():
            _download(src, dst)

    if hf_model_cache := os.getenv("HF_MODEL_CACHE"):
        cache = Path(hf_model_cache)

        if not next((models := bucket / "models").iterdir(), None):
            dprint("Downloading models from S3...")
            _download(models, cache)

        params.model.name = cache / params.model.name
        if params.model.vae:
            params.model.vae = cache / params.model.vae
        if params.model.control_net:
            params.model.control_net = cache / params.model.control_net
        if params.model.refiner:
            params.model.refiner = cache / params.model.refiner
        params.test_model = cache / params.test_model

        _unpack_eval_models(Path(hf_model_cache))

    # _setup_global_cache()

    return {"instance_path": train_data, "params": params}


def standalone_cleanup():
    from cloudpathlib import CloudPath

    if os.getenv("WARM", "0") == "1":
        # dprint("Persisting global cache...")
        # _persist_global_cache()
        dprint("Copying cache back to S3...")
        subprocess.run(
            [
                "rclone",
                "copy",
                "--progress",
                "--checksum",
                "--fast-list",
                "--human-readable",
                "--stats-one-line",
                "--transfers=64",
                "--disable-http2",
                "--checkers=64",
                "--s3-upload-concurrency=8",
                "--s3-env-auth",
                "--s3-region=us-west-2",
                "--s3-use-accelerate-endpoint",
                "--s3-no-check-bucket",
                "--s3-no-head",
                os.environ["CACHE_DIR"],
                cache_path(),
            ],
            capture_output=True,
            check=True,
        )
    else:
        id = os.environ["DREAMBOOTH_ID"]
        bucket = CloudPath(os.environ["DREAMBOOTH_BUCKET"])
        (bucket / "output" / id).upload_from(
            Path("/tmp/ml") / id / "output", force_overwrite_to_cloud=True
        )


def main():
    is_main = os.getenv("LOCAL_RANK", "0") == "0"
    is_sagemaker = "SAGEMAKER_JOB_NAME" in os.environ

    dprint("Starting!", reset=True)

    if is_sagemaker:
        env = environment.Environment()
        env.is_main = is_main
        params = sagemaker_params(env)
        cleanup_fn = partial(sagemaker_cleanup, env)
    elif (id := os.getenv("DREAMBOOTH_ID")) is not None:
        dprint(f"Running standalone job {id}")
        params = standalone_params(is_main)
        cleanup_fn = standalone_cleanup
    else:
        raise ValueError("No input data provided!")

    model = get_model(**params)

    try:
        pipeline = model.train()
        model.eval(pipeline)
    finally:
        if is_main:
            cleanup_fn()
            model.end_training()

        dprint("Exiting!")

    if is_sagemaker:
        sys.exit(0)


if __name__ == "__main__":
    main()