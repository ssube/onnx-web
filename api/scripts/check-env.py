from importlib.metadata import version
from typing import List

REQUIRED_MODULES = ["onnx", "diffusers", "safetensors", "torch"]

RUNTIME_MODULES = [
    "onnxruntime",
    "onnxruntime_gpu",
    "onnxruntime_rocm",
    "onnxruntime_training",
]


def check_modules() -> List[str]:
    results = []
    for name in REQUIRED_MODULES:
        try:
            __import__(name)
            module_version = version(name)
            results.append(
                f"required module {name} is present at version {module_version}"
            )
        except ImportError as e:
            results.append(f"unable to import required module {name}: {e}")

    return results


def check_runtimes() -> List[str]:
    results = []
    for name in RUNTIME_MODULES:
        try:
            __import__(name)
            module_version = version(name)
            results.append(
                f"runtime module {name} is present at version {module_version}"
            )
        except ImportError as e:
            results.append(f"unable to import runtime module {name}: {e}")

    return results


def check_providers() -> List[str]:
    results = []
    try:
        import onnxruntime
        import torch

        available = onnxruntime.get_available_providers()
        for provider in onnxruntime.get_all_providers():
            if provider in available:
                results.append(f"onnxruntime provider {provider} is available")
            else:
                results.append(f"onnxruntime provider {provider} is missing")
    except Exception as e:
        results.append(f"unable to check runtime providers: {e}")

    return results


def check_torch_cuda() -> List[str]:
    results = []
    try:
        import torch.cuda

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    mem_free, mem_total = torch.cuda.mem_get_info()
                    results.append(f"Torch has CUDA device {i} with {mem_free} of {mem_total} bytes of free VRAM")
        else:
            results.append("loaded Torch but CUDA was not available")
    except ImportError as e:
        results.append(f"unable to import Torch CUDA: {e}")
    except Exception as e:
        results.append(f"error listing Torch CUDA: {e}")

    return results


ALL_CHECKS = [
    check_modules,
    check_runtimes,
    check_providers,
    check_torch_cuda,
]


def check_all():
    results = []
    for check in ALL_CHECKS:
        results.extend(check())

    print(results)


if __name__ == "__main__":
    check_all()
