#!/bin/bash
# Build and run the fine-tune container
set -e

echo "Building fine-tune image..."
docker build --network=host -t flan-t5-finetuner ./fine-tune

echo "Running fine-tune container..."
docker run -it --rm \
  -v "$PWD/models:/models" \
  -v "$PWD/data:/data" \
  -e HF_HOME=/models/.cache \
  --name flan-t5-finetuner \
  flan-t5-finetuner

echo "Fine-tuning complete."
