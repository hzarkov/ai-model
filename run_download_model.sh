#!/bin/bash
# Build and run the download-model container as the host user, with model as required argument and optional HF_TOKEN
set -e


if [ -z "$1" ]; then
  echo "Error: No model name provided. Usage: $0 <model-name>"
  exit 1
fi
MODEL_TO_DOWNLOAD="$1"

if [ -z "$HF_TOKEN" ]; then
  echo "Warning: HF_TOKEN is not set. You may need it for gated/private models."
fi

cd "$(dirname "$0")"

echo "Building download-model image..."
docker build --network=host -t model-downloader ./download-model

echo "Running download-model container as UID=$(id -u), GID=$(id -g) for model: $MODEL_TO_DOWNLOAD ..."
docker run --rm -it --network=host \
  -e HOST_UID=$(id -u) \
  -e HOST_GID=$(id -g) \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD/models:/models" \
  --name model-downloader \
  model-downloader "$MODEL_TO_DOWNLOAD"

echo "Model download complete."
