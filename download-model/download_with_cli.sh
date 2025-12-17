#!/bin/bash
set -e

if [ -z "$1" ]; then
	echo "Error: No model name provided. Usage: $0 <model-name>"
	exit 1
fi

MODEL_ID="$1"
OUTPUT_DIR="/models/${MODEL_ID##*/}"

mkdir -p "$OUTPUT_DIR"

# Download the model using the 'hf' CLI
echo "Downloading $MODEL_ID to $OUTPUT_DIR ..."
hf download "$MODEL_ID" --local-dir "$OUTPUT_DIR"

echo "Download complete. Files in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
