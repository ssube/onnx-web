#! /bin/sh

set -eu

if [ -n "${VIRTUAL_ENV+set}" ]; then
  echo "Using current virtual env..."
else
  if [ -d "onnx_env" ]; then
    echo "Loading existing virtual env..."
    . onnx_env/bin/activate
  else
    echo "Creating new virtual env..."
    python -m venv onnx_env
    . onnx_env/bin/activate
  fi
fi

echo "Downloading and converting models to ONNX format..."
python3 -m onnx_web.convert \
  --diffusion \
  --upscaling \
  --correction \
  --extras=${ONNX_WEB_EXTRA_MODELS:-extras.json} \
  --token=${HF_TOKEN:-}

echo "Launching API server..."
flask --app=onnx_web.serve run --host=0.0.0.0
