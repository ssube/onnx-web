from enum import IntEnum
from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .torch_before_ort import GraphOptimizationLevel, SessionOptions

logger = getLogger(__name__)


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

    def with_args(self, **kwargs):
        return Border(
            kwargs.get("left", self.left),
            kwargs.get("right", self.right),
            kwargs.get("top", self.top),
            kwargs.get("bottom", self.bottom),
        )

    @classmethod
    def even(cls, all: int):
        return Border(all, all, all, all)


class Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __iter__(self):
        return iter([self.width, self.height])

    def __str__(self) -> str:
        return "%sx%s" % (self.width, self.height)

    def add_border(self, border: Border):
        return Size(
            border.left + self.width + border.right,
            border.top + self.height + border.bottom,
        )

    def tojson(self) -> Dict[str, int]:
        return {
            "height": self.height,
            "width": self.width,
        }

    def with_args(self, **kwargs):
        return Size(
            kwargs.get("height", self.height),
            kwargs.get("width", self.width),
        )


class DeviceParams:
    def __init__(
        self,
        device: str,
        provider: str,
        options: Optional[dict] = None,
        optimizations: Optional[List[str]] = None,
    ) -> None:
        self.device = device
        self.provider = provider
        self.options = options
        self.optimizations = optimizations or []
        self.sess_options_cache = None

    def __str__(self) -> str:
        return "%s - %s (%s)" % (self.device, self.provider, self.options)

    def ort_provider(self) -> Union[str, Tuple[str, Any]]:
        if self.options is None:
            return self.provider
        else:
            return (self.provider, self.options)

    def sess_options(self) -> SessionOptions:
        if self.sess_options_cache is not None:
            return self.sess_options_cache

        sess = SessionOptions()

        if "onnx-low-memory" in self.optimizations:
            logger.debug("enabling ONNX low-memory optimizations")
            sess.enable_cpu_mem_arena = False
            sess.enable_mem_pattern = False
            sess.enable_mem_reuse = False

        if "onnx-graph-disable" in self.optimizations:
            logger.debug("disabling all ONNX graph optimizations")
            sess.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
        elif "onnx-graph-basic" in self.optimizations:
            logger.debug("enabling basic ONNX graph optimizations")
            sess.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        elif "onnx-graph-all" in self.optimizations:
            logger.debug("enabling all ONNX graph optimizations")
            sess.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        if "onnx-deterministic-compute" in self.optimizations:
            logger.debug("enabling ONNX deterministic compute")
            sess.use_deterministic_compute = True

        self.sess_options_cache = sess
        return sess

    def torch_str(self) -> str:
        if self.device.startswith("cuda"):
            return self.device
        else:
            return "cpu"


class ImageParams:
    def __init__(
        self,
        model: str,
        scheduler: str,
        prompt: str,
        cfg: float,
        steps: int,
        seed: int,
        negative_prompt: Optional[str] = None,
        lpw: bool = False,
        eta: float = 0.0,
        batch: int = 1,
        inversion: Optional[str] = None,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.cfg = cfg
        self.seed = seed
        self.steps = steps
        self.lpw = lpw or False
        self.eta = eta
        self.batch = batch
        self.inversion = inversion

    def tojson(self) -> Dict[str, Optional[Param]]:
        return {
            "model": self.model,
            "scheduler": self.scheduler,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "cfg": self.cfg,
            "seed": self.seed,
            "steps": self.steps,
            "lpw": self.lpw,
            "eta": self.eta,
            "batch": self.batch,
            "inversion": self.inversion,
        }

    def with_args(self, **kwargs):
        return ImageParams(
            kwargs.get("model", self.model),
            kwargs.get("scheduler", self.scheduler),
            kwargs.get("prompt", self.prompt),
            kwargs.get("cfg", self.cfg),
            kwargs.get("steps", self.steps),
            kwargs.get("seed", self.seed),
            kwargs.get("negative_prompt", self.negative_prompt),
            kwargs.get("lpw", self.lpw),
            kwargs.get("eta", self.eta),
            kwargs.get("batch", self.batch),
            kwargs.get("inversion", self.inversion),
        )


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
        upscale_order: Literal[
            "correction-first", "correction-last", "correction-both"
        ] = "correction-first",
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
        self.upscale_order = upscale_order

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
            upscale_order=self.upscale_order,
        )

    def resize(self, size: Size) -> Size:
        face_outscale = self.face_outscale
        if self.upscale_order == "correction-both":
            face_outscale *= self.face_outscale

        return Size(
            size.width * self.outscale * face_outscale,
            size.height * self.outscale * face_outscale,
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
            "upscale_order": self.upscale_order,
        }

    def with_args(self, **kwargs):
        return UpscaleParams(
            kwargs.get("upscale_model", self.upscale_model),
            kwargs.get("correction_model", self.correction_model),
            kwargs.get("denoise", self.denoise),
            kwargs.get("faces", self.faces),
            kwargs.get("face_outscale", self.face_outscale),
            kwargs.get("face_strength", self.face_strength),
            kwargs.get("format", self.format),
            kwargs.get("half", self.half),
            kwargs.get("outscale", self.outscale),
            kwargs.get("pre_pad", self.pre_pad),
            kwargs.get("scale", self.scale),
            kwargs.get("tile_pad", self.tile_pad),
            kwargs.get("upscale_order", self.upscale_order),
        )
