& onnx_env\Scripts\Activate.ps1

echo "Downloading and converting models to ONNX format..."
IF ($Env:ONNX_WEB_EXTRA_MODELS -eq "") {$Env:ONNX_WEB_EXTRA_MODELS="..\models\extras.json"}
python -m onnx_web.convert `
--sources `
--diffusion `
--upscaling `
--correction `
--extras=$Env:ONNX_WEB_EXTRA_MODELS `
--token=$Env:HF_TOKEN $Env:ONNX_WEB_EXTRA_ARGS

echo "Launching API server..."
waitress-serve `
--host=0.0.0.0 `
--port=5000 `
--call `
onnx_web.main:run

pause
