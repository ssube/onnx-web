set ONNX_WEB_BASE_PATH=%~dp0
set ONNX_WEB_BUNDLE_PATH=%ONNX_WEB_BASE_PATH%\client
set ONNX_WEB_MODEL_PATH=%ONNX_WEB_BASE_PATH%\models
set ONNX_WEB_OUTPUT_PATH=%ONNX_WEB_BASE_PATH%\outputs

@echo Launching onnx-web in full-precision mode...
server\onnx-web.exe --diffusion --correction --upscaling