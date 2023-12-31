& onnx_env\Scripts\Activate.ps1

echo "Downloading and converting models to ONNX format..."
IF ($Env:ONNX_WEB_EXTRA_MODELS -eq "") {$Env:ONNX_WEB_EXTRA_MODELS="..\models\extras.json"}
python -m onnx_web.convert `
--sources `
--diffusion `
--upscaling `
--correction `
--networks `
--extras=$Env:ONNX_WEB_EXTRA_MODELS `
--token=$Env:HF_TOKEN $Env:ONNX_WEB_EXTRA_ARGS

if (!(Test-Path -path .\gui\index.html -PathType Leaf)) {
  echo "Downloading latest web UI files from Github..."
  Invoke-WebRequest "https://raw.githubusercontent.com/ssube/onnx-web/gh-pages/v0.11.0/index.html" -OutFile .\gui\index.html
  Invoke-WebRequest "https://raw.githubusercontent.com/ssube/onnx-web/gh-pages/v0.11.0/config.json" -OutFile .\gui\config.json
  Invoke-WebRequest "https://raw.githubusercontent.com/ssube/onnx-web/gh-pages/v0.11.0/bundle/main.js" -OutFile .\gui\bundle\main.js
}

echo "Launching API server..."
waitress-serve `
--host=0.0.0.0 `
--port=5000 `
--call `
onnx_web.main:run

pause
