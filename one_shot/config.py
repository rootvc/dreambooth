import logging
import os
import sys
from pathlib import Path

import diffusers
import torch
import torch._dynamo.config
import torch.backends.cuda
import torch.backends.cudnn
import transformers
from diffusers.utils.logging import disable_progress_bar
from loguru import logger as default_logger
from loguru._logger import Logger

from one_shot.logging import InterceptHandler


def format(record):
    mins = record["elapsed"].seconds // 60
    secs = record["elapsed"].seconds % 60
    rank = record["extra"].get("rank", "MAIN")
    tag = record["extra"].get("tag", "")
    return (
        f"+{mins}m{secs}s"
        + " %s[{level.icon}] \\<%s\\> {message} ({name})\n{exception}" % (rank, tag)
    )


def init_logging(logger: Logger = default_logger):
    logger.remove()

    logger.add(
        sys.stderr,
        level=logging.DEBUG,
        format=format,
        backtrace=True,
        diagnose=True,
        colorize=True,
        enqueue=True,
        context=torch.multiprocessing.get_context("forkserver"),
    )

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def init_torch_config():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 128

    disable_progress_bar()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_warning()


def init_config():
    for env_var in (
        "CACHE_DIR",
        "CACHE_STAGING_DIR",
        "TORCH_HOME",
        "HF_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "PYTORCH_KERNEL_CACHE_PATH",
    ):
        if path := os.getenv(env_var):
            Path(path).mkdir(parents=True, exist_ok=True)
    default_logger.info("Initializing Torch config...")
    init_torch_config()
