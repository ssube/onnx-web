from logging import getLogger
from os import path
from typing import Any, Dict
from zipfile import ZipFile

from regex import match

from .client import fetch_model
from .utils import ConversionContext

logger = getLogger(__name__)


def convert_extract_archive(
    conversion: ConversionContext, model: Dict[str, Any], format: str
):
    name = str(model.get("name")).strip()
    source = model.get("source")

    dest_path = path.join(conversion.model_path, name)

    logger.info("extracting archived model %s: %s -> %s/", name, source, dest_path)

    if path.exists(dest_path):
        logger.info("destination path already exists, skipping extraction")
        return False, dest_path

    cache_path = fetch_model(conversion, name, model["source"], format=format)

    with ZipFile(cache_path) as zip:
        names = zip.namelist()
        if not all([is_safe(name) for name in names]):
            raise ValueError("archive contains unsafe filenames")

        logger.debug("archive is valid, extracting all files: %s", names)
        zip.extractall(path=dest_path)
        return True, dest_path


SAFE_NAME = r"^[-_a-zA-Z/\\\.]+$"


def is_safe(name: str):
    return match(SAFE_NAME, name)
