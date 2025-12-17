# Dockerfile for downloading google/flan-t5-large model using huggingface-cli
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git gosu && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir huggingface_hub

# Create the models directory
RUN mkdir -p /models

COPY download_with_cli.sh .
COPY entrypoint.sh .

VOLUME ["/models"]

ENV HOST_UID=1000
ENV HOST_GID=1000

ENTRYPOINT ["bash", "entrypoint.sh"]
