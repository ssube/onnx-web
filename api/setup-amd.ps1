python -m venv onnx_env
& onnx_env\Scripts\Activate.ps1

pip install -r requirements/amd-windows.txt
pip install -r requirements/base.txt

.\launch.ps1