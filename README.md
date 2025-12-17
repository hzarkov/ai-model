
# FLAN-T5 Model Download & Fine-tune Pipeline

This project provides Docker containers and scripts for downloading and fine-tuning FLAN-T5 models (any size).

## Structure

```
.
├── download-model/          # Container for downloading the model (huggingface-cli based)
│   ├── Dockerfile
│   ├── entrypoint.sh
│   ├── download_with_cli.sh
├── fine-tune/               # Container for fine-tuning the model
│   ├── Dockerfile
│   ├── fine_tune.py
│   └── requirements.txt
├── data/                    # Training data directory
│   └── training_data.json   # Sample training data
├── models/                  # Model storage (created automatically)
├── run_download_model.sh    # Script to build/run the download-model image
├── run_fine_tune.sh         # Script to build/run the fine-tune image
└── README.md
```

## Usage


### 1. Download a FLAN-T5 Model

To download a model (default: google/flan-t5-large):

```bash
bash run_download_model.sh
```

To download a different model (e.g., google/flan-t5-small):

```bash
bash run_download_model.sh google/flan-t5-small
```

This will build and run the download-model container with host networking, saving the model to `./models/<model-name>`.

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

This will build and run the fine-tune container, saving the fine-tuned model to `./models/flan-t5-large-finetuned` (or similar).

## Notes
- Edit `data/training_data.json` to customize your training data.
- Both scripts mount the `models` and `data` directories for persistence.
- Host networking is used for the download step to avoid connectivity issues.
- You can specify any model from the Hugging Face Hub as an argument to the download script.
