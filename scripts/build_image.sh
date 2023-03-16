#!/bin/bash
set -euxo pipefail

export DOCKER_BUILDKIT=1

init() {
  git pull --rebase
  aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 630351220487.dkr.ecr.us-west-2.amazonaws.com
  docker buildx create --use --name builder \
    --buildkitd-flags '--oci-worker-snapshotter=stargz' \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SIZE=-1 \
    --driver-opt env.BUILDKIT_STEP_LOG_MAX_SPEED=-1 ||
    true
  docker buildx inspect --bootstrap builder
}

pull_build_push() {
  DOCKERFILE="dockerfiles/$1"
  IMAGE_NAME="rootventures/$2:latest"
  docker pull "$IMAGE_NAME"
  docker build \
    -o type=registry,oci-mediatypes=true \
    --cache-from=type=registry,ref=rootventures/$2:cache \
    --cache-to=type=registry,ref=rootventures/$2:cache,mode=max \
    --builder builder \
    -f "$DOCKERFILE" . \
    -t "$IMAGE_NAME"
  docker pull "$IMAGE_NAME"
}

init

pull_build_push Dockerfile.pytorch pytorch
pull_build_push Dockerfile.train train-dreambooth
pull_build_push Dockerfile.sagemaker train-dreambooth-sagemaker
pull_build_push Dockerfile.runpod train-dreambooth-runpod

docker build \
  -o type=registry,oci-mediatypes=true,compression=zstd \
  --cache-from=type=registry,ref=rootventures/train-dreambooth-sagemaker:cache \
  --builder builder \
  -f dockerfiles/Dockerfile.sagemaker . \
  -t 630351220487.dkr.ecr.us-west-2.amazonaws.com/train-dreambooth-sagemaker:latest

# pull_build_push Dockerfile dreambooth

docker system prune -f
