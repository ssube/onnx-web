import sys
import traceback
from io import BytesIO
from logging import getLogger
from logging.config import dictConfig
from os import environ, path
from time import sleep
from typing import Optional

import cv2
import numpy as np
import requests
from PIL import Image
from yaml import safe_load

TEST_DATA = [
    (
        "txt2img-sd-v1-5-256-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&width=256&height=256",
    ),
    (
        "txt2img-sd-v1-5-512-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim",
    ),
    (
        "txt2img-sd-v2-1-512-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1",
    ),
    (
        "txt2img-sd-v2-1-768-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1&width=768&height=768",
    ),
]

logging_path = environ.get("ONNX_WEB_LOGGING_PATH", "./logging.yaml")

try:
    if path.exists(logging_path):
        with open(logging_path, "r") as f:
            config_logging = safe_load(f)
            dictConfig(config_logging)
except Exception as err:
    print("error loading logging config: %s" % (err))

logger = getLogger(__name__)


def test_root() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    else:
        return "http://127.0.0.1:5000"


def test_path(relpath: str) -> str:
    return path.join(path.dirname(__file__), relpath)


def generate_image(root: str, params: str) -> Optional[str]:
    resp = requests.post(f"{root}/api/{params}")
    if resp.status_code == 200:
        json = resp.json()
        return json.get("output")
    else:
        logger.warning("request failed: %s", resp.status_code)
        return None


def check_ready(root: str, key: str) -> bool:
    resp = requests.get(f"{root}/api/ready?output={key}")
    if resp.status_code == 200:
        json = resp.json()
        return json.get("ready", False)
    else:
        logger.warning("request failed: %s", resp.status_code)
        return False


def download_image(root: str, key: str) -> Image.Image:
    resp = requests.get(f"{root}/output/{key}")
    if resp.status_code == 200:
        return Image.open(BytesIO(resp.content))
    else:
        logger.warning("request failed: %s", resp.status_code)
        return None


def find_mse(result: Image.Image, ref: Image.Image) -> float:
    if result.mode != ref.mode:
        logger.warning("image mode does not match: %s vs %s", result.mode, ref.mode)
        return float("inf")

    if result.size != ref.size:
        logger.warning("image size does not match: %s vs %s", result.size, ref.size)
        return float("inf")

    nd_result = np.array(result)
    nd_ref = np.array(ref)

    diff = cv2.subtract(nd_ref, nd_result)
    diff = np.sum(diff**2)

    return diff / (float(ref.height * ref.width)) / 255.0


def run_test(
    root: str,
    name: str,
    params: str,
    ref: Image.Image,
    max_attempts: int = 20,
    mse_threshold: float = 0.0001,
) -> bool:
    """
    Generate an image, wait for it to be ready, and calculate the MSE from the reference.
    """

    logger.info("running test: %s", params)

    key = generate_image(root, params)
    if key is None:
        raise ValueError("could not generate")

    attempts = 0
    while attempts < max_attempts and not check_ready(root, key):
        logger.debug("waiting for image to be ready")
        sleep(6)

    if attempts == max_attempts:
        raise ValueError("image was not ready in time")

    result = download_image(root, key)
    result.save(test_path(path.join("test-results", f"{name}.png")))
    mse = find_mse(result, ref)

    if mse < mse_threshold:
        logger.debug("MSE within threshold: %.4f < %.4f", mse, mse_threshold)
        return True
    else:
        logger.warning("MSE above threshold: %.4f > %.4f", mse, mse_threshold)
        return False


def main():
    root = test_root()
    logger.info("running release tests against API: %s", root)

    failures = 0
    for name, query in TEST_DATA:
        try:
            ref_name = test_path(path.join("test-refs", f"{name}.png"))
            ref = Image.open(ref_name) if path.exists(ref_name) else None
            if run_test(root, name, query, ref):
                logger.info("test passed: %s", name)
            else:
                logger.warning("test failed: %s", name)
                failures += 1
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            logger.error("error running test for %s: %s", name, e)
            failures += 1

    if failures > 0:
        logger.error("%s tests had errors", failures)
        sys.exit(1)

if __name__ == "__main__":
    main()
