REM paths required to make the bundle work
set ONNX_WEB_BASE_PATH=%~dp0
set ONNX_WEB_BUNDLE_PATH=%ONNX_WEB_BASE_PATH%\client
set ONNX_WEB_MODEL_PATH=%ONNX_WEB_BASE_PATH%\models
set ONNX_WEB_OUTPUT_PATH=%ONNX_WEB_BASE_PATH%\outputs

REM customize these as needed
set ONNX_WEB_BLOCK_PLATFORMS=cpu
set ONNX_WEB_CACHE_MODELS=0
set ONNX_WEB_EXTRA_MODELS=%ONNX_WEB_BASE_PATH%\models\extras.json

REM convert models and launch the server
@echo Launching onnx-web in fp16 mode...
server\onnx-web.exe --diffusion --correction --upscaling --half