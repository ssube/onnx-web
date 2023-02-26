echo "Downloading and converting models to ONNX format..."
python -m onnx_web.convert --sources --diffusion --upscaling --correction --token=%HF_TOKEN%

echo "Launching API server..."
flask --app="onnx_web.serve:run" run --host=0.0.0.0
