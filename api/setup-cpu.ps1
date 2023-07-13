python -m venv onnx_env
& onnx_env\Scripts\Activate.ps1

pip install -r requirements/cpu.txt
pip install -r requirements/base.txt

.\launch.ps1
