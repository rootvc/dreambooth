#!/bin/bash
set -euxo pipefail
cd /dreambooth

export PYTHONPATH=$PYTHONPATH:.

git pull origin master --rebase

accelerate launch dreambooth/train/train.py
