#!/bin/bash
# Script to download pre-compiled HEF model from GitHub Releases

# TODO: Update this URL after creating the GitHub Release
# Example: https://github.com/DanielDubinsky/yolo26_hailo/releases/download/v0.1.0/yolo26n.hef
MODEL_URL="https://github.com/DanielDubinsky/yolo26_hailo/releases/latest/download/yolo26n.hef"
OUTPUT_DIR="$(dirname "$0")/../models"
OUTPUT_FILE="$OUTPUT_DIR/yolo26n.hef"

mkdir -p "$OUTPUT_DIR"

echo "Downloading yolo26n.hef from GitHub Releases..."
if wget -O "$OUTPUT_FILE" "$MODEL_URL"; then
    echo "Successfully downloaded to $OUTPUT_FILE"
else
    echo "Error downloading model. Please check the URL or network connection."
    echo "URL attempted: $MODEL_URL"
    exit 1
fi
