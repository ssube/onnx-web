echo "Downloading and converting models to ONNX format..."
python -m onnx_web.convert --diffusion --upscaling --correction --extras --token=%HF_TOKEN%

echo "Launching API server..."
flask --app=onnx_web.serve run --host=0.0.0.0
