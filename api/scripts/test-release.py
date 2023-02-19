import sys
import traceback
from io import BytesIO
from logging import getLogger
from logging.config import dictConfig
from os import environ, path
from time import sleep
from typing import Optional
from collections import Counter

import cv2
import numpy as np
import requests
from PIL import Image
from yaml import safe_load

class TestCase:
    def __init__(
        self,
        name: str,
        query: str,
        max_attempts: int = 20,
        mse_threshold: float = 0.0001,
    ) -> None:
        self.name = name
        self.query = query
        self.max_attempts = max_attempts
        self.mse_threshold = mse_threshold


TEST_DATA = [
    TestCase(
        "txt2img-sd-v1-5-256-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&width=256&height=256",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-deis",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=deis",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-dpm",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=dpm-multi",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-heun",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=heun",
    ),
    TestCase(
        "txt2img-sd-v2-1-512-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1",
    ),
    TestCase(
        "txt2img-sd-v2-1-768-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1&width=768&height=768",
    ),
    TestCase(
        "txt2img-openjourney-512-muffin",
        "txt2img?prompt=mdjrny-v4+style+a+giant+muffin&seed=0&scheduler=ddim&model=diffusion-openjourney",
    ),
    TestCase(
        "txt2img-knollingcase-512-muffin",
        "txt2img?prompt=knollingcase+display+case+with+a+giant+muffin&seed=0&scheduler=ddim&model=diffusion-knollingcase",
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
    test: TestCase,
    ref: Image.Image,
) -> bool:
    """
    Generate an image, wait for it to be ready, and calculate the MSE from the reference.
    """

    logger.info("running test: %s", test.query)

    key = generate_image(root, test.query)
    if key is None:
        raise ValueError("could not generate")

    attempts = 0
    while attempts < test.max_attempts and not check_ready(root, key):
        logger.debug("waiting for image to be ready")
        sleep(6)

    if attempts == test.max_attempts:
        raise ValueError("image was not ready in time")

    result = download_image(root, key)
    result.save(test_path(path.join("test-results", f"{test.name}.png")))
    mse = find_mse(result, ref)

    if mse < test.mse_threshold:
        logger.debug("MSE within threshold: %.4f < %.4f", mse, test.mse_threshold)
        return True
    else:
        logger.warning("MSE above threshold: %.4f > %.4f", mse, test.mse_threshold)
        return False


def main():
    root = test_root()
    logger.info("running release tests against API: %s", root)

    results = Counter({
        True: 0,
        False: 0,
    })
    for test in TEST_DATA:
        try:
            ref_name = test_path(path.join("test-refs", f"{test.name}.png"))
            ref = Image.open(ref_name) if path.exists(ref_name) else None
            if run_test(root, test, ref):
                logger.info("test passed: %s", test.name)
                results[True] += 1
            else:
                logger.warning("test failed: %s", test.name)
                results[False] += 1
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            logger.error("error running test for %s: %s", test.name, e)
            results[False] += 1

    logger.info("%s of %s tests passed", results[True], results[True] + results[False])
    if results[False] > 0:
        logger.error("%s tests had errors", results[False])
        sys.exit(1)

if __name__ == "__main__":
    main()
