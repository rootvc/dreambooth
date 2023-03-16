#!/bin/bash
set -euxo pipefail

export PYTHONPATH=${PYTHONPATH-}:.

echo $(nvidia-smi -L | wc -l) >>/root/.cache/huggingface/accelerate/default_config.yaml

python -m dreambooth.warm
