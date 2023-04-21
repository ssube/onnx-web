echo "Downloading and converting models to ONNX format..."
IF "%ONNX_WEB_EXTRA_MODELS%"=="" (set ONNX_WEB_EXTRA_MODELS=..\models\extras.json)
python -m onnx_web.convert ^
--sources ^
--diffusion ^
--upscaling ^
--correction ^
--extras=%ONNX_WEB_EXTRA_MODELS% ^
--token=%HF_TOKEN% %ONNX_WEB_EXTRA_ARGS%

echo "Launching API server..."
waitress-serve ^
--host=0.0.0.0 ^
--port=5000 ^
--call ^
onnx_web.main:run

pause
