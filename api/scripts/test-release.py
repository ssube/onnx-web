import sys
from argparse import ArgumentParser
from io import BytesIO
from logging import getLogger
from logging.config import dictConfig
from os import environ, path
from time import sleep
from tqdm import tqdm
from typing import List, Optional, Union

import cv2
import numpy as np
import requests
from PIL import Image
from yaml import safe_load

logging_path = environ.get("ONNX_WEB_LOGGING_PATH", "./logging.yaml")

try:
    if path.exists(logging_path):
        with open(logging_path, "r") as f:
            config_logging = safe_load(f)
            dictConfig(config_logging)
except Exception as err:
    print("error loading logging config: %s" % (err))

logger = getLogger(__name__)

FAST_TEST = 10
SLOW_TEST = 25
VERY_SLOW_TEST = 100

STRICT_TEST = 1e-4
LOOSE_TEST = 1e-2
VERY_LOOSE_TEST = 0.025


def test_path(relpath: str) -> str:
    return path.join(path.dirname(__file__), relpath)


class TestCase:
    def __init__(
        self,
        name: str,
        query: str,
        max_attempts: int = FAST_TEST,
        mse_threshold: float = STRICT_TEST,
        source: Union[Image.Image, List[Image.Image]] = None,
        mask: Image.Image = None,
    ) -> None:
        self.name = name
        self.query = query
        self.max_attempts = max_attempts
        self.mse_threshold = mse_threshold
        self.source = source
        self.mask = mask


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
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-dpm",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=dpm-multi",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-heun",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=heun",
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-unipc",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=unipc-multi",
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "txt2img-sd-v2-1-512-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1",
    ),
    TestCase(
        "txt2img-sd-v2-1-768-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v2-1&width=768&height=768&unet_tile=768",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "txt2img-openjourney-256-muffin",
        "txt2img?prompt=mdjrny-v4+style+a+giant+muffin&seed=0&scheduler=ddim&model=diffusion-openjourney&width=256&height=256",
    ),
    TestCase(
        "txt2img-openjourney-512-muffin",
        "txt2img?prompt=mdjrny-v4+style+a+giant+muffin&seed=0&scheduler=ddim&model=diffusion-openjourney",
    ),
    TestCase(
        "txt2img-knollingcase-512-muffin",
        "txt2img?prompt=knollingcase+display+case+with+a+giant+muffin&seed=0&scheduler=ddim&model=diffusion-knollingcase",
    ),
    TestCase(
        "img2img-sd-v1-5-512-pumpkin",
        "img2img?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&sourceFilter=none",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "img2img-sd-v1-5-256-pumpkin",
        "img2img?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&sourceFilter=none&unet_tile=256",
        source="txt2img-sd-v1-5-256-muffin-0",
    ),
    TestCase(
        "inpaint-v1-512-white",
        "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting",
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-white",
    ),
    TestCase(
        "inpaint-v1-512-black",
        "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting",
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
    ),
    TestCase(
        "outpaint-even-256",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=fill-mask"
            "&top=256&bottom=256&left=256&right=256"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=SLOW_TEST,
        mse_threshold=VERY_LOOSE_TEST,
    ),
    TestCase(
        "outpaint-vertical-512",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=fill-mask"
            "&top=512&bottom=512&left=0&right=0"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=SLOW_TEST,
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "outpaint-horizontal-512",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=fill-mask"
            "&top=0&bottom=0&left=512&right=512"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=SLOW_TEST,
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "upscale-resrgan-x2-1024-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-real-esrgan-x2-plus&scale=2&outscale=2&upscale=true",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "upscale-resrgan-x4-2048-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-real-esrgan-x4-plus&scale=4&outscale=4&upscale=true",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "blend-512-muffin-black",
        "blend?prompt=a+giant+pumpkin&seed=0&scheduler=ddim",
        mask="mask-black",
        source=[
            "txt2img-sd-v1-5-512-muffin-0",
            "txt2img-sd-v2-1-512-muffin-0",
        ],
    ),
    TestCase(
        "blend-512-muffin-white",
        "blend?prompt=a+giant+pumpkin&seed=0&scheduler=ddim",
        mask="mask-white",
        source=[
            "txt2img-sd-v1-5-512-muffin-0",
            "txt2img-sd-v2-1-512-muffin-0",
        ],
    ),
    TestCase(
        "blend-512-muffin-blend",
        "blend?prompt=a+giant+pumpkin&seed=0&scheduler=ddim",
        mask="mask-blend",
        source=[
            "txt2img-sd-v1-5-512-muffin-0",
            "txt2img-sd-v2-1-512-muffin-0",
        ],
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-taters",
        "txt2img?prompt=<lora:taters:1.0>+a+giant+muffin+made+of+mashed+potatoes&seed=0&scheduler=unipc-multi",
    ),
    TestCase(
        "txt2img-sd-v1-5-512-muffin-cloud",
        "txt2img?prompt=<inversion:cloud:1.0>+a+giant+muffin+made+of+cloud-all&seed=0&scheduler=unipc-multi",
    ),
    TestCase(
        "upscale-swinir-x4-2048-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-swinir-real-large-x4&scale=4&outscale=4&upscale=true",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "upscale-codeformer-512-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&correction=correction-codeformer&faces=true&faceOutscale=1&faceStrength=1.0",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "upscale-gfpgan-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=correction-gfpgan&faces=true&faceOutscale=1&faceStrength=1.0",
        source="txt2img-sd-v1-5-512-muffin-0",
    ),
    TestCase(
        "upscale-sd-x4-2048-muffin",
        "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-stable-diffusion-x4&scale=4&outscale=4&upscale=true",
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=VERY_SLOW_TEST,
    ),
    TestCase(
        "outpaint-panorama-even-256",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=fill-mask"
            "&top=256&bottom=256&left=256&right=256&pipeline=panorama"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=VERY_SLOW_TEST,
        mse_threshold=VERY_LOOSE_TEST,
    ),
    TestCase(
        "outpaint-panorama-vertical-512",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=histogram"
            "&top=512&bottom=512&left=0&right=0&pipeline=panorama"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=VERY_SLOW_TEST,
        mse_threshold=VERY_LOOSE_TEST,
    ),
    TestCase(
        "outpaint-panorama-horizontal-512",
        (
            "inpaint?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&model=stable-diffusion-onnx-v1-inpainting&noise=histogram"
            "&top=0&bottom=0&left=512&right=512&pipeline=panorama"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        mask="mask-black",
        max_attempts=VERY_SLOW_TEST,
        mse_threshold=VERY_LOOSE_TEST,
    ),
    TestCase(
        "upscale-resrgan-x4-codeformer-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-real-esrgan-x4-plus&scale=4&outscale=4"
            "&correction=correction-codeformer&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "upscale-resrgan-x4-gfpgan-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-real-esrgan-x4-plus&scale=4&outscale=4"
            "&correction=correction-gfpgan&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "upscale-swinir-x4-codeformer-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-swinir-real-large-x4&scale=4&outscale=4"
            "&correction=correction-codeformer&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "upscale-swinir-x4-gfpgan-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-swinir-real-large-x4&scale=4&outscale=4"
            "&correction=correction-gfpgan&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "upscale-sd-x4-codeformer-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-stable-diffusion-x4&scale=4&outscale=4"
            "&correction=correction-codeformer&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=VERY_SLOW_TEST,
    ),
    TestCase(
        "upscale-sd-x4-gfpgan-2048-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-stable-diffusion-x4"
            "&scale=4&outscale=4&correction=correction-gfpgan&faces=true&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-512-muffin-0",
        max_attempts=VERY_SLOW_TEST,
    ),
    TestCase(
        "txt2img-panorama-1024x768-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&width=1024&height=768&pipeline=panorama&tiled_vae=true",
        max_attempts=VERY_SLOW_TEST,
    ),
    TestCase(
        "img2img-panorama-1024x768-pumpkin",
        "img2img?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&sourceFilter=none&pipeline=panorama&tiled_vae=true",
        source="txt2img-panorama-1024x768-muffin-0",
        max_attempts=VERY_SLOW_TEST,
    ),
    TestCase(
        "txt2img-sd-v1-5-tall-muffin",
        "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&width=512&height=768&unet_tile=768",
    ),
    TestCase(
        "upscale-resrgan-x4-tall-muffin",
        (
            "upscale?prompt=a+giant+pumpkin&seed=0&scheduler=ddim&upscaling=upscaling-real-esrgan-x4-plus"
            "&scale=4&outscale=4&correction=correction-gfpgan&faces=false&faceOutscale=1&faceStrength=1.0&upscale=true"
        ),
        source="txt2img-sd-v1-5-tall-muffin-0",
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "txt2img-sdxl-muffin",
        (
            "txt2img?prompt=a+giant+muffin&seed=0&scheduler=ddim&width=1024&height=1024&unet_tile=1024"
            "&pipeline=txt2img-sdxl&model=diffusion-sdxl-base"
        ),
        max_attempts=SLOW_TEST,
    ),
    TestCase(
        "txt2img-sdxl-lcm-muffin",
        (
            "txt2img?prompt=<lora:sdxl-lcm:1.0>+a+giant+muffin&seed=0&scheduler=lcm&width=1024&height=1024"
            "&unet_tile=1024&pipeline=txt2img-sdxl&model=diffusion-sdxl-base&cfg=1.5&steps=10"
        ),
        max_attempts=SLOW_TEST,
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "txt2img-sdxl-turbo-muffin",
        (
            "txt2img?prompt=a+giant+muffin&seed=0&scheduler=dpm-sde&width=512&height=512&unet_tile=512"
            "&pipeline=txt2img-sdxl&model=diffusion-sdxl-turbo&cfg=1&steps=5"
        ),
        max_attempts=SLOW_TEST,
        mse_threshold=LOOSE_TEST,
    ),
    TestCase(
        "txt2img-sd-v1-5-lcm-muffin",
        (
            "txt2img?prompt=<lora:lcm:1.0>+a+giant+muffin&seed=0&scheduler=lcm&width=512&height=512&unet_tile=512"
            "&pipeline=txt2img&cfg=1.5&steps=10"
        ),
        max_attempts=SLOW_TEST,
        mse_threshold=VERY_LOOSE_TEST,
    ),
    # TODO: non-square controlnet
]


class TestError(Exception):
    def __str__(self) -> str:
        return super().__str__()


class TestResult:
    error: Optional[str]
    mse: Optional[float]
    name: str
    passed: bool

    def __init__(self, name: str, error = None, passed = True, mse = None) -> None:
        self.error = error
        self.mse = mse
        self.name = name
        self.passed = passed

    def __repr__(self) -> str:
        if self.passed:
            if self.mse is not None:
                return f"{self.name} ({self.mse})"
            else:
                return self.name
        else:
            if self.mse is not None:
                return f"{self.name}: {self.error} ({self.mse})"
            else:
                return f"{self.name}: {self.error}"

    @classmethod
    def passed(self, name: str, mse = None):
        return TestResult(name, mse=mse)

    @classmethod
    def failed(self, name: str, error: str, mse = None):
        return TestResult(name, error=error, mse=mse, passed=False)


def parse_args(args: List[str]):
    parser = ArgumentParser(
        prog="onnx-web release tests",
        description="regression tests for onnx-web",
    )
    parser.add_argument("--host", default="http://127.0.0.1:5000")
    parser.add_argument("-n", "--name", help="filter tests by name (contains this string)")
    parser.add_argument("-m", "--mse", default=1.0, type=float, help="MSE multiplier (test strictness)")
    parser.add_argument("-t", "--time", default=1, type=int, help="time multiplier (test duration)")
    return parser.parse_args(args)


def generate_images(host: str, test: TestCase) -> Optional[str]:
    files = {}
    if test.source is not None:
        if isinstance(test.source, list):
            for i in range(len(test.source)):
                source = test.source[i]
                logger.debug("loading test source %s: %s", i, source)
                source_path = test_path(path.join("test-refs", f"{source}.png"))
                source_image = Image.open(source_path)
                source_bytes = BytesIO()
                source_image.save(source_bytes, "png")
                source_bytes.seek(0)
                files[f"source:{i}"] = source_bytes

        else:
            logger.debug("loading test source: %s", test.source)
            source_path = test_path(path.join("test-refs", f"{test.source}.png"))
            source_image = Image.open(source_path)
            source_bytes = BytesIO()
            source_image.save(source_bytes, "png")
            source_bytes.seek(0)
            files["source"] = source_bytes

    if test.mask is not None:
        logger.debug("loading test mask: %s", test.mask)
        mask_path = test_path(path.join("test-refs", f"{test.mask}.png"))
        mask_image = Image.open(mask_path)
        mask_bytes = BytesIO()
        mask_image.save(mask_bytes, "png")
        mask_bytes.seek(0)
        files["mask"] = mask_bytes

    logger.debug("generating image: %s", test.query)
    resp = requests.post(f"{host}/api/{test.query}", files=files)
    if resp.status_code == 200:
        json = resp.json()
        return json.get("outputs")
    else:
        logger.warning("generate request failed: %s: %s", resp.status_code, resp.text)
        raise TestError("error generating image")


def check_ready(host: str, key: str) -> bool:
    resp = requests.get(f"{host}/api/ready?output={key}")
    if resp.status_code == 200:
        json = resp.json()
        ready = json.get("ready", False)
        if ready:
            cancelled = json.get("cancelled", False)
            failed = json.get("failed", False)
            return not cancelled and not failed
        else:
            return False
    else:
        logger.warning("ready request failed: %s", resp.status_code)
        raise TestError("error getting image status")


def download_images(host: str, keys: List[str]) -> List[Image.Image]:
    images = []
    for key in keys:
        resp = requests.get(f"{host}/output/{key}")
        if resp.status_code == 200:
            logger.debug("downloading image: %s", key)
            images.append(Image.open(BytesIO(resp.content)))
        else:
            logger.warning("download request failed: %s", resp.status_code)
            raise TestError("error downloading image")

    return images


def find_mse(result: Image.Image, ref: Image.Image) -> float:
    if result.mode != ref.mode:
        logger.warning("image mode does not match: %s vs %s", result.mode, ref.mode)
        return float("inf")

    if result.size != ref.size:
        logger.warning("image size does not match: %s vs %s", result.size, ref.size)
        return float("inf")

    nd_result = np.array(result)
    nd_ref = np.array(ref)

    # dividing before squaring reduces the error into the lower end of the [0, 1] range
    diff = cv2.subtract(nd_ref, nd_result) / 255.0
    diff = np.sum(diff**2)

    return diff / (float(ref.height * ref.width))


def run_test(
    host: str,
    test: TestCase,
    mse_mult: float = 1.0,
    time_mult: int = 1,
) -> TestResult:
    """
    Generate an image, wait for it to be ready, and calculate the MSE from the reference.
    """

    keys = generate_images(host, test)
    if keys is None:
        return TestResult.failed(test.name, "could not generate image")

    ready = False
    for attempt in tqdm(range(test.max_attempts * time_mult)):
        if check_ready(host, keys[0]):
            logger.debug("image is ready: %s", keys)
            ready = True
            break
        else:
            logger.debug("waiting for image to be ready")
            sleep(6)

    if not ready:
        return TestResult.failed(test.name, "image was not ready in time")

    results = download_images(host, keys)
    if results is None or len(results) == 0:
        return TestResult.failed(test.name, "could not download image")

    passed = False
    for i in range(len(results)):
        result = results[i]
        result.save(test_path(path.join("test-results", f"{test.name}-{i}.png")))

        ref_name = test_path(path.join("test-refs", f"{test.name}-{i}.png"))
        ref = Image.open(ref_name) if path.exists(ref_name) else None

        mse = find_mse(result, ref)
        threshold = test.mse_threshold * mse_mult

        if mse < threshold:
            logger.info("MSE within threshold: %.5f < %.5f", mse, threshold)
            passed = True
        else:
            logger.warning("MSE above threshold: %.5f > %.5f", mse, threshold)
            return TestResult.failed(test.name, error="MSE above threshold", mse=mse)

    if passed:
        return TestResult.passed(test.name)
    else:
        return TestResult.failed(test.name, "no images tested")


def main():
    args = parse_args(sys.argv[1:])
    logger.info("running release tests against API: %s", args.host)

    if args.name is None:
        tests = TEST_DATA
    else:
        tests = [test for test in TEST_DATA if args.name in test.name]

    # make sure tests have unique names
    test_names = [test.name for test in tests]
    if len(test_names) != len(set(test_names)):
        logger.error("tests must have unique names: %s", test_names)
        sys.exit(1)

    passed = []
    failed = []
    for test in tests:
        result = None

        for _i in range(3):
            try:
                logger.info("starting test: %s", test.name)
                result = run_test(args.host, test, mse_mult=args.mse, time_mult=args.time)
                if result.passed:
                    logger.info("test passed: %s", test.name)
                    break
                else:
                    logger.warning("test failed: %s", test.name)
            except Exception:
                logger.exception("error running test for %s", test.name)
                result = TestResult.failed(test.name, "TODO: exception message")

        if result is not None:
            if result.passed:
                passed.append(result)
            else:
                failed.append(result)

    logger.info("%s of %s tests passed", len(passed), len(tests))
    failed = list(set(failed))
    if len(failed) > 0:
        logger.error("%s tests had errors: %s", len(failed), failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
