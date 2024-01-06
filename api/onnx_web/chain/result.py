from json import dumps
from logging import getLogger
from os import path
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..convert.utils import resolve_tensor
from ..params import Border, HighresParams, ImageParams, Size, UpscaleParams
from ..server.context import ServerContext
from ..server.load import get_extra_hashes
from ..utils import hash_file

logger = getLogger(__name__)


class NetworkMetadata:
    name: str
    hash: str
    weight: float

    def __init__(self, name: str, hash: str, weight: float) -> None:
        self.name = name
        self.hash = hash
        self.weight = weight


class ImageMetadata:
    ancestors: List["ImageMetadata"]
    params: ImageParams
    size: Size

    # models
    inversions: List[NetworkMetadata]
    loras: List[NetworkMetadata]
    models: List[NetworkMetadata]

    # optional params
    border: Optional[Border]
    highres: Optional[HighresParams]
    upscale: Optional[UpscaleParams]

    @staticmethod
    def unknown_image() -> "ImageMetadata":
        UNKNOWN_STR = "unknown"
        return ImageMetadata(
            ImageParams(UNKNOWN_STR, UNKNOWN_STR, UNKNOWN_STR, "", 0, 0, 0),
            Size(0, 0),
        )

    def __init__(
        self,
        params: ImageParams,
        size: Size,
        upscale: Optional[UpscaleParams] = None,
        border: Optional[Border] = None,
        highres: Optional[HighresParams] = None,
        inversions: Optional[List[NetworkMetadata]] = None,
        loras: Optional[List[NetworkMetadata]] = None,
        models: Optional[List[NetworkMetadata]] = None,
        ancestors: Optional[List["ImageMetadata"]] = None,
    ) -> None:
        self.params = params
        self.size = size
        self.upscale = upscale
        self.border = border
        self.highres = highres
        self.inversions = inversions or []
        self.loras = loras or []
        self.models = models or []
        self.ancestors = ancestors or []

    def child(
        self,
        params: ImageParams,
        size: Size,
        upscale: Optional[UpscaleParams] = None,
        border: Optional[Border] = None,
        highres: Optional[HighresParams] = None,
        inversions: Optional[List[NetworkMetadata]] = None,
        loras: Optional[List[NetworkMetadata]] = None,
        models: Optional[List[NetworkMetadata]] = None,
    ) -> "ImageMetadata":
        return ImageMetadata(
            params,
            size,
            upscale,
            border,
            highres,
            inversions,
            loras,
            models,
            [self],
        )

    def get_model_hash(
        self, server: ServerContext, model: Optional[str] = None
    ) -> Tuple[str, str]:
        model_name = path.basename(path.normpath(model or self.params.model))
        logger.debug("getting model hash for %s", model_name)

        if model_name in server.hash_cache:
            logger.debug("using cached model hash for %s", model_name)
            return (model_name, server.hash_cache[model_name])

        model_hash = get_extra_hashes().get(model_name, None)
        if model_hash is None:
            model_hash_path = path.join(self.params.model, "hash.txt")
            if path.exists(model_hash_path):
                with open(model_hash_path, "r") as f:
                    model_hash = f.readline().rstrip(",. \n\t\r")

        model_hash = model_hash or "unknown"
        server.hash_cache[model_name] = model_hash

        return (model_name, model_hash)

    def get_network_hash(
        self, server: ServerContext, network_name: str, network_type: str
    ) -> Tuple[str, str]:
        # run this again just in case the file path changes
        network_path = resolve_tensor(
            path.join(server.model_path, network_type, network_name)
        )

        if network_path in server.hash_cache:
            logger.debug("using cached network hash for %s", network_path)
            return (network_name, server.hash_cache[network_path])

        network_hash = hash_file(network_path).upper()
        server.hash_cache[network_path] = network_hash

        return (network_name, network_hash)

    def to_exif(self, server: ServerContext, output: List[str]) -> str:
        model_name, model_hash = self.get_model_hash(server)
        hash_map = {
            model_name: model_hash,
        }

        inversion_hashes = ""
        if self.inversions is not None:
            inversion_pairs = [
                (
                    name,
                    self.get_network_hash(server, name, "inversion")[1],
                )
                for name, _weight in self.inversions
            ]
            inversion_hashes = ",".join(
                [f"{name}: {hash}" for name, hash in inversion_pairs]
            )
            hash_map.update(dict(inversion_pairs))

        lora_hashes = ""
        if self.loras is not None:
            lora_pairs = [
                (
                    name,
                    self.get_network_hash(server, name, "lora")[1],
                )
                for name, _weight in self.loras
            ]
            lora_hashes = ",".join([f"{name}: {hash}" for name, hash in lora_pairs])
            hash_map.update(dict(lora_pairs))

        return (
            f"{self.params.prompt or ''}\nNegative prompt: {self.params.negative_prompt or ''}\n"
            f"Steps: {self.params.steps}, Sampler: {self.params.scheduler}, CFG scale: {self.params.cfg}, "
            f"Seed: {self.params.seed}, Size: {self.size.width}x{self.size.height}, "
            f"Model hash: {model_hash}, Model: {model_name}, "
            f"Tool: onnx-web, Version: {server.server_version}, "
            f'Inversion hashes: "{inversion_hashes}", '
            f'Lora hashes: "{lora_hashes}", '
            f"Hashes: {dumps(hash_map)}"
        )

    def tojson(self, server: ServerContext, output: List[str]):
        json = {
            "input_size": self.size.tojson(),
            "outputs": output,
            "params": self.params.tojson(),
            "inversions": [],
            "loras": [],
            "models": [],
        }

        # fix up some fields
        model_name, model_hash = self.get_model_hash(server, self.params.model)
        json["params"]["model"] = model_name
        json["models"].append(
            {
                "hash": model_hash,
                "name": model_name,
                "weight": 1.0,
            }
        )

        # calculate final output size
        output_size = self.size
        if self.border is not None:
            json["border"] = self.border.tojson()
            output_size = output_size.add_border(self.border)

        if self.highres is not None:
            json["highres"] = self.highres.tojson()
            output_size = self.highres.resize(output_size)

        if self.upscale is not None:
            json["upscale"] = self.upscale.tojson()
            output_size = self.upscale.resize(output_size)

        json["size"] = output_size.tojson()

        if self.inversions is not None:
            for name, weight in self.inversions:
                hash = self.get_network_hash(server, name, "inversion")[1]
                json["inversions"].append(
                    {"name": name, "weight": weight, "hash": hash}
                )

        if self.loras is not None:
            for name, weight in self.loras:
                hash = self.get_network_hash(server, name, "lora")[1]
                json["loras"].append({"name": name, "weight": weight, "hash": hash})

        if self.models is not None:
            for name, weight in self.models:
                name, hash = self.get_model_hash()
                json["models"].append({"name": name, "weight": weight, "hash": hash})

        return json


class StageResult:
    """
    Chain pipeline stage result.
    Can contain PIL images or numpy arrays, with helpers to convert between them.
    This class intentionally does not provide `__iter__`, to ensure clients get results in the format
    they are expected.
    """

    arrays: Optional[List[np.ndarray]]
    images: Optional[List[Image.Image]]
    metadata: List[ImageMetadata]

    @staticmethod
    def empty():
        return StageResult(images=[])

    @staticmethod
    def from_arrays(arrays: List[np.ndarray], metadata: List[ImageMetadata]):
        return StageResult(arrays=arrays, metadata=metadata)

    @staticmethod
    def from_images(images: List[Image.Image], metadata: List[ImageMetadata]):
        return StageResult(images=images, metadata=metadata)

    def __init__(
        self,
        arrays: Optional[List[np.ndarray]] = None,
        images: Optional[List[Image.Image]] = None,
        metadata: Optional[List[ImageMetadata]] = None,  # TODO: should not be optional
        source: Optional[Any] = None,
    ) -> None:
        data_provided = sum(
            [arrays is not None, images is not None, source is not None]
        )
        if data_provided > 1:
            raise ValueError("results must only contain one type of data")
        elif data_provided == 0:
            raise ValueError("results must contain some data")

        if source is not None:
            self.arrays = source.arrays
            self.images = source.images
            self.metadata = source.metadata
        else:
            self.arrays = arrays
            self.images = images
            self.metadata = metadata or []

    def __len__(self) -> int:
        if self.arrays is not None:
            return len(self.arrays)
        elif self.images is not None:
            return len(self.images)
        else:
            return 0

    def as_arrays(self) -> List[np.ndarray]:
        if self.arrays is not None:
            return self.arrays
        elif self.images is not None:
            return [np.array(i) for i in self.images]
        else:
            return []

    def as_images(self) -> List[Image.Image]:
        if self.images is not None:
            return self.images
        elif self.arrays is not None:
            return [Image.fromarray(np.uint8(i), shape_mode(i)) for i in self.arrays]
        else:
            return []

    def push_array(self, array: np.ndarray, metadata: ImageMetadata):
        if self.arrays is not None:
            self.arrays.append(array)
        elif self.images is not None:
            self.images.append(Image.fromarray(np.uint8(array), shape_mode(array)))
        else:
            self.arrays = [array]

        if metadata is not None:
            self.metadata.append(metadata)
        else:
            raise ValueError("metadata must be provided")

    def push_image(self, image: Image.Image, metadata: ImageMetadata):
        if self.images is not None:
            self.images.append(image)
        elif self.arrays is not None:
            self.arrays.append(np.array(image))
        else:
            self.images = [image]

        if metadata is not None:
            self.metadata.append(metadata)
        else:
            raise ValueError("metadata must be provided")

    def insert_array(self, index: int, array: np.ndarray, metadata: ImageMetadata):
        if self.arrays is not None:
            self.arrays.insert(index, array)
        elif self.images is not None:
            self.images.insert(
                index, Image.fromarray(np.uint8(array), shape_mode(array))
            )
        else:
            self.arrays = [array]

        if metadata is not None:
            self.metadata.insert(index, metadata)
        else:
            raise ValueError("metadata must be provided")

    def insert_image(self, index: int, image: Image.Image, metadata: ImageMetadata):
        if self.images is not None:
            self.images.insert(index, image)
        elif self.arrays is not None:
            self.arrays.insert(index, np.array(image))
        else:
            self.images = [image]

        if metadata is not None:
            self.metadata.insert(index, metadata)
        else:
            raise ValueError("metadata must be provided")

    def size(self) -> Size:
        if self.images is not None:
            return Size(
                max([image.width for image in self.images], default=0),
                max([image.height for image in self.images], default=0),
            )
        elif self.arrays is not None:
            return Size(
                max([array.shape[0] for array in self.arrays], default=0),
                max([array.shape[1] for array in self.arrays], default=0),
            )  # TODO: which fields within the shape are width/height?
        else:
            return Size(0, 0)

    def validate(self) -> None:
        """
        Make sure the data exists and that data and metadata match in length.
        """

        if self.arrays is None and self.images is None:
            raise ValueError("no data in result")

        if len(self) != len(self.metadata):
            raise ValueError("metadata and data do not match in length")


def shape_mode(arr: np.ndarray) -> str:
    if len(arr.shape) != 3:
        raise ValueError("unknown array format")

    if arr.shape[-1] == 3:
        return "RGB"
    elif arr.shape[-1] == 4:
        return "RGBA"

    raise ValueError("unknown image format")
