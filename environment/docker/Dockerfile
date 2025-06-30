# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY . /workspace/

RUN pip3 install --no-cache-dir -r requirements.txt || true

CMD ["bash"] 