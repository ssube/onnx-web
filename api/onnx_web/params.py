from enum import IntEnum
from logging import getLogger
from math import ceil
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .models.meta import NetworkModel
from .torch_before_ort import GraphOptimizationLevel, SessionOptions

logger = getLogger(__name__)


Param = Union[str, int, float]
Point = Tuple[int, int]


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


class Border:
    def __init__(self, left: int, right: int, top: int, bottom: int) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __str__(self) -> str:
        return "(%s, %s, %s, %s)" % (self.left, self.right, self.top, self.bottom)

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

    def round_to_tile(self, tile = 512):
        return Size(
            ceil(self.width / tile) * tile,
            ceil(self.height / tile) * tile,
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

    def ort_provider(
        self, model_type: Optional[str] = None
    ) -> Union[str, Tuple[str, Any]]:
        if model_type is not None:
            # check if model has been pinned to CPU
            # TODO: check whether the CPU device is allowed
            if f"onnx-cpu-{model_type}" in self.optimizations:
                return "CPUExecutionProvider"

        if self.options is None:
            return self.provider
        else:
            return (self.provider, self.options)

    def sess_options(self, cache=True) -> SessionOptions:
        if cache and self.sess_options_cache is not None:
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

        if cache:
            self.sess_options_cache = sess

        return sess

    def torch_str(self) -> str:
        if self.device.startswith("cuda"):
            if self.options is not None and "device_id" in self.options:
                return f"{self.device}:{self.options['device_id']}"

            return self.device
        elif self.device.startswith("rocm"):
            if self.options is not None and "device_id" in self.options:
                return f"cuda:{self.options['device_id']}"

            return "cuda"
        else:
            return "cpu"


class ImageParams:
    model: str
    pipeline: str
    scheduler: str
    prompt: str
    cfg: float
    steps: int
    seed: int
    negative_prompt: Optional[str]
    lpw: bool
    eta: float
    batch: int
    control: Optional[NetworkModel]
    input_prompt: str
    input_negative_prompt: str
    loopback: int

    def __init__(
        self,
        model: str,
        pipeline: str,
        scheduler: str,
        prompt: str,
        cfg: float,
        steps: int,
        seed: int,
        negative_prompt: Optional[str] = None,
        eta: float = 0.0,
        batch: int = 1,
        control: Optional[NetworkModel] = None,
        input_prompt: Optional[str] = None,
        input_negative_prompt: Optional[str] = None,
        loopback: int = 0,
    ) -> None:
        self.model = model
        self.pipeline = pipeline
        self.scheduler = scheduler
        self.prompt = prompt
        self.cfg = cfg
        self.steps = steps
        self.seed = seed
        self.negative_prompt = negative_prompt
        self.eta = eta
        self.batch = batch
        self.control = control
        self.input_prompt = input_prompt or prompt
        self.input_negative_prompt = input_negative_prompt or negative_prompt
        self.loopback = loopback

    def do_cfg(self):
        return self.cfg > 1.0

    def get_valid_pipeline(self, group: str, pipeline: str = None) -> str:
        pipeline = pipeline or self.pipeline

        # if the correct pipeline was already requested, simply use that
        if group == pipeline:
            return pipeline

        # otherwise, check for additional allowed pipelines
        if group == "img2img":
            if pipeline in ["controlnet", "lpw", "panorama", "pix2pix"]:
                return pipeline
        elif group == "inpaint":
            if pipeline in ["controlnet"]:
                return pipeline
        elif group == "txt2img":
            if pipeline in ["panorama"]:
                return pipeline

        logger.debug("pipeline %s is not valid for %s", pipeline, group)
        return group

    def lpw(self):
        return self.pipeline == "lpw"

    def tojson(self) -> Dict[str, Optional[Param]]:
        return {
            "model": self.model,
            "pipeline": self.pipeline,
            "scheduler": self.scheduler,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "cfg": self.cfg,
            "seed": self.seed,
            "steps": self.steps,
            "eta": self.eta,
            "batch": self.batch,
            "control": self.control.name if self.control is not None else "",
            "input_prompt": self.input_prompt,
            "input_negative_prompt": self.input_negative_prompt,
            "loopback": self.loopback,
        }

    def with_args(self, **kwargs):
        return ImageParams(
            kwargs.get("model", self.model),
            kwargs.get("pipeline", self.pipeline),
            kwargs.get("scheduler", self.scheduler),
            kwargs.get("prompt", self.prompt),
            kwargs.get("cfg", self.cfg),
            kwargs.get("steps", self.steps),
            kwargs.get("seed", self.seed),
            kwargs.get("negative_prompt", self.negative_prompt),
            kwargs.get("eta", self.eta),
            kwargs.get("batch", self.batch),
            kwargs.get("control", self.control),
            kwargs.get("input_prompt", self.input_prompt),
            kwargs.get("input_negative_prompt", self.input_negative_prompt),
            kwargs.get("loopback", self.loopback),
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
        format: Literal["onnx", "pth"] = "onnx",  # TODO: deprecated, remove
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
            kwargs.get("outscale", self.outscale),
            kwargs.get("scale", self.scale),
            kwargs.get("pre_pad", self.pre_pad),
            kwargs.get("tile_pad", self.tile_pad),
            kwargs.get("upscale_order", self.upscale_order),
        )


class HighresParams:
    def __init__(
        self,
        scale: int,
        steps: int,
        strength: float,
        method: Literal["bilinear", "lanczos", "upscale"] = "lanczos",
        iterations: int = 1,
        tiled_vae: bool = False,
    ):
        self.scale = scale
        self.steps = steps
        self.strength = strength
        self.method = method
        self.iterations = iterations
        self.tiled_vae = tiled_vae

    def resize(self, size: Size) -> Size:
        return Size(
            size.width * (self.scale**self.iterations),
            size.height * (self.scale**self.iterations),
        )

    def tojson(self):
        return {
            "iterations": self.iterations,
            "method": self.method,
            "scale": self.scale,
            "steps": self.steps,
            "strength": self.strength,
            "tiled_vae": self.tiled_vae,
        }
