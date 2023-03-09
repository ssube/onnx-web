echo "Downloading and converting models to ONNX format..."
IF "%ONNX_WEB_EXTRA_MODELS%"=="" (set ONNX_WEB_EXTRA_MODELS=extras.json)
python -m onnx_web.convert ^
--sources ^
--diffusion ^
--upscaling ^
--correction ^
--extras=%ONNX_WEB_EXTRA_MODELS% ^
--token=%HF_TOKEN% ^
%ONNX_WEB_EXTRA_ARGS%

echo "Launching API server..."
flask --app="onnx_web.main:run" run --host=0.0.0.0
