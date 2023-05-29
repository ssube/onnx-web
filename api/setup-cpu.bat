python -m venv onnx_env
call onnx_env\Scripts\Activate.bat

pip install -r requirements/cpu.txt
pip install -r requirements/base.txt

.\launch.bat