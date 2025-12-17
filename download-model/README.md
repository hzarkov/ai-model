# Download google/flan-t5-large with huggingface-cli

This Docker image downloads the google/flan-t5-large model using the lightweight huggingface-cli tool.

## Usage

Build the image:

```bash
docker build --network=host -t flan-t5-downloader-cli ./download-model-cli
```

Run the container:

```bash
docker run --rm --network=host -v "$PWD/models:/models" --name flan-t5-downloader-cli flan-t5-downloader-cli
```

The model will be saved to `./models/flan-t5-large` on your host.
