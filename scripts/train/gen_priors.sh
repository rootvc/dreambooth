#!/bin/bash
set -euxo pipefail
cd /dreambooth

if [ ! -d ".git" ]; then
  git init .
  git remote add origin https://github.com/rootvc/dreambooth.git
fi

git fetch origin main
git reset --hard origin/main

export ACCELERATE_LOG_LEVEL=info
export DIFFUSERS_VERBOSITY=info
export PYTHONPATH=${PYTHONPATH-}:.

python -m dreambooth.train.gen_priors
