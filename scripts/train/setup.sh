#!/bin/bash
set -euxo pipefail

echo $(nvidia-smi -L | wc -l) >>/root/.cache/huggingface/accelerate/default_config.yaml

echo "export AWS_DEFAULT_REGION=us-west-2" >>/root/.bashrc
echo "export AWS_ACCESS_KEY_ID=$(gum input --placeholder AWS_ACCESS_KEY_ID)" >>/root/.bashrc
echo "export AWS_SECRET_ACCESS_KEY=$(gum input --placeholder AWS_SECRET_ACCESS_KEY)" >>/root/.bashrc
