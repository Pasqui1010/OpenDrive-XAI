#!/usr/bin/env bash
# download_sample_data.sh — fetch ~50 MB mini dataset for quick demos
set -euo pipefail

DATA_DIR="$(git rev-parse --show-toplevel)/data/sample"
mkdir -p "$DATA_DIR"

URL="https://opendrive-xai-public.s3.amazonaws.com/sample_dataset_v0.1.zip"
FILE="sample_dataset_v0.1.zip"

if [ -f "$DATA_DIR/$FILE" ]; then
  echo "Sample dataset already downloaded. Skipping."
  exit 0
fi

echo "Downloading sample dataset (≈50 MB)…"
curl -L "$URL" -o "$DATA_DIR/$FILE"

echo "Unzipping…"
unzip -q "$DATA_DIR/$FILE" -d "$DATA_DIR"
rm "$DATA_DIR/$FILE"

echo "Done. Files in $DATA_DIR" 