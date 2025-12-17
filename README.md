# Model Download, Fine-tune & Serve Pipeline

This project provides Docker containers and scripts for downloading, fine-tuning, and serving transformer models (any size/type).

## Structure

```
.
├── download-model/          # Container for downloading models (huggingface-cli based)
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── download_with_cli.sh
├── fine-tune/               # Container for fine-tuning models
│   ├── Dockerfile
│   ├── fine_tune.py
├── serve-api/               # Container for serving models via FastAPI
│   ├── Dockerfile
│   ├── app.py
│   └── README.md
├── data/                    # Training data directory
│   └── training_data.json   # Sample training data
├── models/                  # Model storage (created automatically)
├── run_download_model.sh    # Script to build/run the model-downloader image
├── run_fine_tune.sh         # Script to build/run the model-finetuner image
├── run_serve_api.sh         # Script to build/run the model-api image
└── README.md
```

## Usage

### 1. Download a Model

You must specify the model name (e.g., google/flan-t5-large):

```bash
bash run_download_model.sh google/flan-t5-large
```

This will build and run the model-downloader container, saving the model to `./models/<model-name>`.

### 2. Prepare Your Training Data

Edit `data/training_data.json` with your training examples. The format should be:

```json
[
  {
    "input": "Your input text here",
    "output": "Expected output text here"
  },
  ...
]
```

### 3. Fine-tune the Model

```bash
bash run_fine_tune.sh
```

This will build and run the model-finetuner container, saving the fine-tuned model to `./models/<model-name>-finetuned` (or similar).

### 4. Serve the Model via API

You must specify the model directory to serve (e.g., flan-t5-large-finetuned):

```bash
bash run_serve_api.sh flan-t5-large-finetuned
```

This will build and run the model-api container, serving the model at http://localhost:8000.

## Notes
- All scripts use generic Docker image/container names (model-downloader, model-finetuner, model-api).
- All scripts require the model name or directory as an argument where appropriate.
- Both scripts mount the `models` and `data` directories for persistence.
- Host networking is used for the download step to avoid connectivity issues.
- You can specify any model from the Hugging Face Hub as an argument to the download script.
- The containers run as your host user for correct file permissions.
- For gated/private models, set the HF_TOKEN environment variable before running the download script.
