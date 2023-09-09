#!/bin/bash
set -euxo pipefail
cd /dreambooth

if [ ! -d ".git" ]; then
  git init .
  git remote add origin https://github.com/rootvc/dreambooth.git
fi

git fetch origin main
git reset --hard origin/main

export TORCHDYNAMO_VERBOSE=1
export ACCELERATE_LOG_LEVEL=info
export PYTHONPATH=${PYTHONPATH-}:.

export GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "num_processes: $GPU_COUNT" >>/root/.cache/huggingface/accelerate/default_config.yaml

python -m dreambooth.runpod
