#!/bin/bash
set -euxo pipefail

export DOCKER_BUILDKIT=1

build_push() {
  DOCKERFILE="dockerfiles/$1"
  IMAGE_NAME="rootventures/$2"

  depot build \
    --pull \
    --platform linux/amd64 \
    -f "$DOCKERFILE" . \
    -t "$IMAGE_NAME:latest" \
    -t "$IMAGE_NAME:main" \
    --cache-from=type=registry,ref=rootventures/$2:cache \
    -o "type=registry,oci-mediatypes=true,compression=zstd,force-compression=true,compression-level=5,name-canonical=true"
}

build_push Dockerfile.pytorch pytorch min
build_push Dockerfile.train train-dreambooth min
build_push Dockerfile.runpod train-dreambooth-runpod min
