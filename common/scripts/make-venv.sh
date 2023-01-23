#! /bin/sh

if [ -d onnx_env ];
then
  echo "Activating existing venv..."
  source onnx_env/bin/activate
else
  echo "Creating new venv..."
  python3 -m venv onnx_env
fi
