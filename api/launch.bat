call onnx_env\Scripts\Activate.bat

echo "This launch.bat script is deprecated in favor of launch.ps1 and will be removed in a future release."

echo "Downloading and converting models to ONNX format..."
IF "%ONNX_WEB_EXTRA_MODELS%"=="" (set ONNX_WEB_EXTRA_MODELS=..\models\extras.json)
python -m onnx_web.convert ^
--correction ^
--diffusion ^
--networks ^
--sources ^
--upscaling ^
--extras=%ONNX_WEB_EXTRA_MODELS% ^
--token=%HF_TOKEN% %ONNX_WEB_EXTRA_ARGS%

IF NOT EXIST .\gui\index.html (
  echo "Please make sure you have downloaded the web UI files from https://github.com/ssube/onnx-web/tree/gh-pages"
  echo "See https://github.com/ssube/onnx-web/blob/main/docs/setup-guide.md#download-the-web-ui-bundle for more information"
  pause
)

echo "Launching API server..."
waitress-serve ^
--host=0.0.0.0 ^
--port=5000 ^
--call ^
onnx_web.main:run

pause
