#! /bin/sh

set -eu

echo "Downloading and converting models to ONNX format..."
python3 -m onnx_web.convert --diffusion --upscaling --correction --extras=./extras.json --token=${HF_TOKEN:-}

echo "Launching API server..."
flask --app=onnx_web.serve run --host=0.0.0.0
