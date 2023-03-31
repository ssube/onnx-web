python -m venv onnx_env
onnx_env\Scripts\Activate.bat

pip install -r requirements/nvidia.txt
pip install -r requirements/base.txt

.\launch.bat