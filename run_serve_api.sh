#!/bin/bash
# Build and run the FastAPI serving container for FLAN-T5 models
set -e

cd "$(dirname "$0")"


if [ -z "$1" ]; then
  echo "Error: No model name provided. Usage: $0 <model-name>"
  exit 1
fi
MODEL_NAME="$1"
MODEL_PATH="/models/$MODEL_NAME"
PORT="${PORT:-8000}"

# Build the image

echo "Building FastAPI serving image..."
docker build --network=host -t flan-t5-api ./serve-api

echo "Running FastAPI server for model at $MODEL_PATH ..."
docker run -it --rm -p $PORT:8000 \
  -v "$PWD/models:/models" \
  -e MODEL_PATH="$MODEL_PATH" \
  --name flan-t5-api \
  flan-t5-api
