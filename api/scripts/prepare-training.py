from argparse import ArgumentParser
from typing import Any, List, Tuple
from PIL.Image import Image, open as pil_open, merge, Resampling
from torchvision.transforms import RandomCrop, Resize, Normalize, ToTensor
from os import environ, path
from logging import getLogger
from logging.config import dictConfig
from yaml import safe_load
from glob import iglob
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
from PIL import ImageOps


logging_path = environ.get("ONNX_WEB_LOGGING_PATH", "./logging.yaml")

try:
    if path.exists(logging_path):
        with open(logging_path, "r") as f:
            config_logging = safe_load(f)
            dictConfig(config_logging)
except Exception as err:
    print("error loading logging config: %s" % (err))

logger = getLogger(__name__)


def parse_args():
    parser = ArgumentParser()

    # paths
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)

    # image params
    parser.add_argument("--crops", type=int)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.75)

    return parser.parse_args()


def load_images(root: str) -> List[Tuple[str, Image]]:
    logger.info("loading images from %s", root)

    images = []
    for name in iglob(path.join(root, '**', '*.jpg'), recursive=True):
        logger.info("loading image file: %s", name)
        prefix, _ext = path.splitext(name)
        prefix = path.basename(prefix)

        try:
            image = pil_open(name)
            image = ImageOps.exif_transpose(image)

            if image.mode == "L":
                image = merge("RGB", (image, image, image))

            logger.info("adding %s to sources", name)
            images.append((prefix, image))
        except:
            logger.exception("error loading image")

    return images


def save_images(root: str, images: List[Tuple[str, Image]]):
    for name, image in images:
        logger.info("saving image %s", name)
        image.save(path.join(root, f"crop_{name}.jpg"))

    logger.info("saved %s images to %s", len(images), root)


def resize_images(images: List[Tuple[str, Image]], size: Tuple[int, int], min_scale: float) -> List[Tuple[str, Image]]:
    results = []
    for name, image in images:
        scale = min(image.width / size[0], image.height / size[1])
        resize = (int(image.width / scale), int(image.height / scale))
        logger.info("resize %s from %s to %s (%s scale)", name, image.size, resize, scale)

        if scale < min_scale:
            logger.warning("image %s is too small: %s", name, resize)
            continue

        results.append((name, image.resize(resize, Resampling.LANCZOS)))

    return results


def remove_duplicates(sources: List[Tuple[str, Image]], threshold: float, vector_cache: List[Any]) -> List[Tuple[str, Image]]:
    model = models.resnet18(pretrained=True)
    model.eval()

    # prepare transforms to make images resnet-compatible
    scaler = Resize((224, 224))
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensor()

    vectors = []
    for name, source in sources:
        source_tensor = Variable(normalize(to_tensor(scaler(source))).unsqueeze(0))
        vectors.append(model(source_tensor))

    similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    results = []
    for (name, source), source_vector in zip(sources, vectors):
        cached = False
        for cache_vector in vector_cache:
            score = similarity(source_vector, cache_vector)
            logger.debug("similarity score for %s: %s", name, score)

            if score.max() > threshold:
                cached = True

        if cached == False:
            vector_cache.append(source_vector)
            results.append((name, source))

    logger.info("keeping %s of %s images", len(results), len(sources))

    return results


def crop_images(sources: List[Tuple[str, Image]], size: Tuple[int, int], crops: int) -> List[Tuple[str, Image]]:
    transform = RandomCrop(size)
    results = []

    for name, source in sources:
        logger.info("cropping %s", name)

        if source.width < size[0] or source.height < size[1]:
            logger.info("a small image leaked into the set: %s", name)
            continue

        for i in range(crops):
            results.append((f"{name}_{i}", transform(source)))

    return results


if __name__ == "__main__":
    args = parse_args()
    size = (int(args.width * args.scale), int(args.height * args.scale))

    # load unique sources
    sources = load_images(args.src)
    logger.info("loaded %s source images, resizing", len(sources))
    sources = resize_images(sources, size, 0.5)
    logger.info("resized images, removing duplicates")
    sources = remove_duplicates(sources, args.threshold, [])
    logger.info("removed duplicated, kept %s source images", len(sources))

    # randomly crop
    cache = []
    count = 0
    for source in sources:
        crops = crop_images([source], (args.width, args.height), args.crops)
        crops = remove_duplicates(crops, args.threshold, cache)
        save_images(args.dest, crops)
        count += len(crops)

    logger.info("saved %s crops from %s sources", count, len(sources))
