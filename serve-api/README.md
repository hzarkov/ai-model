# FLAN-T5 FastAPI Serving Container

This container serves a FLAN-T5 model (any size) via a FastAPI REST API using ROCm/PyTorch for AMD GPUs.

## Usage

1. Build the image:

```bash
docker build -t flan-t5-api ./serve-api
```

2. Run the container (mount your model directory):

```bash
docker run --rm -p 8000:8000 -v "$PWD/models:/models" -e MODEL_PATH=/models/flan-t5-large-finetuned flan-t5-api
```

- Change `MODEL_PATH` to the directory of your model if needed.
- The API will be available at http://localhost:8000

## Endpoints

- `POST /generate` — Generate text from input
  - JSON body: `{ "input_text": "...", "max_new_tokens": 64, "temperature": 1.0 }`
  - Returns: `{ "result": "..." }`

- `GET /` — Health check
