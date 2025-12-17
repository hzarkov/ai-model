#!/bin/bash
# Build and run the fine-tune container
set -e


# Get host UID and GID
HOST_UID=$(id -u)
HOST_GID=$(id -g)

echo "Building fine-tune image..."
docker build --network=host --build-arg HOST_UID=$HOST_UID --build-arg HOST_GID=$HOST_GID -t model-finetuner ./fine-tune


echo "Running fine-tune container..."
docker run -it --rm \
  -v "$PWD/models:/models" \
  -v "$PWD/data:/data" \
  -e HF_HOME=/models/.cache \
  -e HOST_UID=$HOST_UID \
  -e HOST_GID=$HOST_GID \
  --name model-finetuner \
  model-finetuner

echo "Fine-tuning complete."
