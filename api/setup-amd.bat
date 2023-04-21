python -m venv onnx_env
onnx_env\Scripts\Activate.bat

pip install -r requirements/amd-windows.txt
pip install -r requirements/base.txt

.\launch.bat