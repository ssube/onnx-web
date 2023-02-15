from enum import IntEnum
from typing import Any, Dict, Literal, Optional, Tuple, Union

from onnxruntime import SessionOptions


class SizeChart(IntEnum):
    mini = 128  # small tile for very expensive models
    half = 256  # half tile for outpainting
    auto = 512  # auto tile size
    hd1k = 2**10
    hd2k = 2**11
    hd4k = 2**12
    hd8k = 2**13
    hd16k = 2**14
    hd64k = 2**16


class TileOrder:
    grid = "grid"
    kernel = "kernel"
    spiral = "spiral"


Param = Union[str, int, float]
Point = Tuple[int, int]


class Border:
    def __init__(self, left: int, right: int, top: int, bottom: int) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __str__(self) -> str:
        return "%s %s %s %s" % (self.left, self.top, self.right, self.bottom)

    def tojson(self):
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }

    @classmethod
    def even(cls, all: int):
        return Border(all, all, all, all)


class Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return "%sx%s" % (self.width, self.height)

    def add_border(self, border: Border):
        return Size(
            border.left + self.width + border.right,
            border.top + self.height + border.right,
        )

    def tojson(self) -> Dict[str, int]:
        return {
            "height": self.height,
            "width": self.width,
        }


class DeviceParams:
    def __init__(
        self, device: str, provider: str, options: Optional[dict] = None
    ) -> None:
        self.device = device
        self.provider = provider
        self.options = options

    def __str__(self) -> str:
        return "%s - %s (%s)" % (self.device, self.provider, self.options)

    def ort_provider(self) -> Tuple[str, Any]:
        if self.options is None:
            return self.provider
        else:
            return (self.provider, self.options)

    def sess_options(self) -> SessionOptions:
        return SessionOptions()

    def torch_device(self) -> str:
        if self.device.startswith("cuda"):
            return self.device
        else:
            return "cpu"


class ImageParams:
    def __init__(
        self,
        model: str,
        scheduler: Any,
        prompt: str,
        cfg: float,
        steps: int,
        seed: int,
        negative_prompt: Optional[str] = None,
        lpw: Optional[bool] = False,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.cfg = cfg
        self.seed = seed
        self.steps = steps
        self.lpw = lpw or False

    def tojson(self) -> Dict[str, Optional[Param]]:
        return {
            "model": self.model,
            "scheduler": self.scheduler.__name__,
            "prompt": self.prompt,
            "negativePrompt": self.negative_prompt,
            "cfg": self.cfg,
            "seed": self.seed,
            "steps": self.steps,
            "lpw": self.lpw,
        }


class StageParams:
    """
    Parameters for a chained pipeline stage
    """

    def __init__(
        self,
        name: Optional[str] = None,
        outscale: int = 1,
        tile_order: str = TileOrder.grid,
        tile_size: int = SizeChart.auto,
        # batch_size: int = 1,
    ) -> None:
        self.name = name
        self.outscale = outscale
        self.tile_order = tile_order
        self.tile_size = tile_size


class UpscaleParams:
    def __init__(
        self,
        upscale_model: str,
        correction_model: Optional[str] = None,
        denoise: float = 0.5,
        faces=True,
        face_outscale: int = 1,
        face_strength: float = 0.5,
        format: Literal["onnx", "pth"] = "onnx",
        half=False,
        outscale: int = 1,
        scale: int = 4,
        pre_pad: int = 0,
        tile_pad: int = 10,
    ) -> None:
        self.upscale_model = upscale_model
        self.correction_model = correction_model
        self.denoise = denoise
        self.faces = faces
        self.face_outscale = face_outscale
        self.face_strength = face_strength
        self.format = format
        self.half = half
        self.outscale = outscale
        self.pre_pad = pre_pad
        self.scale = scale
        self.tile_pad = tile_pad

    def rescale(self, scale: int):
        return UpscaleParams(
            self.upscale_model,
            correction_model=self.correction_model,
            denoise=self.denoise,
            faces=self.faces,
            face_outscale=self.face_outscale,
            face_strength=self.face_strength,
            format=self.format,
            half=self.half,
            outscale=scale,
            scale=scale,
            pre_pad=self.pre_pad,
            tile_pad=self.tile_pad,
        )

    def resize(self, size: Size) -> Size:
        return Size(
            size.width * self.outscale * self.face_outscale,
            size.height * self.outscale * self.face_outscale,
        )

    def tojson(self):
        return {
            "upscale_model": self.upscale_model,
            "correction_model": self.correction_model,
            "denoise": self.denoise,
            "faces": self.faces,
            "face_outscale": self.face_outscale,
            "face_strength": self.face_strength,
            "format": self.format,
            "half": self.half,
            "outscale": self.outscale,
            "pre_pad": self.pre_pad,
            "scale": self.scale,
            "tile_pad": self.tile_pad,
        }
