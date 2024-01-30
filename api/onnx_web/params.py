from enum import IntEnum
from logging import getLogger
from math import ceil
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .models.meta import NetworkModel
from .torch_before_ort import GraphOptimizationLevel, SessionOptions
from .utils import coalesce

logger = getLogger(__name__)


Param = Union[str, int, float]
Point = Tuple[int, int]


class SizeChart(IntEnum):
    micro = 64
    mini = 128  # small tile for very expensive models
    half = 256  # half tile for outpainting
    auto = 512  # auto tile size
    hd1k = 2**10
    hd2k = 2**11
    hd4k = 2**12
    hd8k = 2**13
    hd16k = 2**14
    hd32k = 2**15
    hd64k = 2**16
    max = 2**32  # should be a reasonable upper limit for now


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

    def isZero(self) -> bool:
        return all(
            value == 0 for value in (self.left, self.right, self.top, self.bottom)
        )

    def tojson(self):
        return {
            "left": self.left,
            "right": self.right,
            "top": self.top,
            "bottom": self.bottom,
        }

    def with_args(
        self,
        left: Optional[int] = None,
        right: Optional[int] = None,
        top: Optional[int] = None,
        bottom: Optional[int] = None,
        **kwargs,
    ):
        logger.debug("ignoring extra kwargs for border: %s", kwargs)
        return Border(
            left or self.left,
            right or self.right,
            top or self.top,
            bottom or self.bottom,
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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Size):
            return self.width == other.width and self.height == other.height

        return False

    def add_border(self, border: Border):
        return Size(
            border.left + self.width + border.right,
            border.top + self.height + border.bottom,
        )

    def max(self, width: int, height: int):
        return Size(max(self.width, width), max(self.height, height))

    def min(self, width: int, height: int):
        return Size(min(self.width, width), min(self.height, height))

    def round_to_tile(self, tile=512):
        return Size(
            ceil(self.width / tile) * tile,
            ceil(self.height / tile) * tile,
        )

    def tojson(self) -> Dict[str, int]:
        return {
            "width": self.width,
            "height": self.height,
        }

    def with_args(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs,
    ):
        logger.debug("ignoring extra kwargs for size: %s", kwargs)
        return Size(
            width or self.width,
            height or self.height,
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
        self,
        model_type: str,
        suffix: Optional[str] = None,
    ) -> Union[str, Tuple[str, Any]]:
        # check if model has been pinned to CPU
        # TODO: check whether the CPU device is allowed
        if f"onnx-cpu-{model_type}" in self.optimizations:
            logger.debug("pinning %s to CPU", model_type)
            return "CPUExecutionProvider"

        if (
            suffix is not None
            and f"onnx-cpu-{model_type}-{suffix}" in self.optimizations
        ):
            logger.debug("pinning %s-%s to CPU", model_type, suffix)
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
        elif "onnx-graph-extended" in self.optimizations:
            logger.debug("enabling extended ONNX graph optimizations")
            sess.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_EXTENDED
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
    eta: float
    batch: int
    control: Optional[NetworkModel]
    input_prompt: str
    input_negative_prompt: Optional[str]
    loopback: int
    tiled_vae: bool
    unet_tile: int
    unet_overlap: float
    vae_tile: int
    vae_overlap: float
    denoise: int
    thumbnail: int

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
        tiled_vae: bool = False,
        unet_overlap: float = 0.25,
        unet_tile: int = 512,
        vae_overlap: float = 0.25,
        vae_tile: int = 512,
        denoise: int = 3,
        thumbnail: int = 1,
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
        self.tiled_vae = tiled_vae
        self.unet_overlap = unet_overlap
        self.unet_tile = unet_tile
        self.vae_overlap = vae_overlap
        self.vae_tile = vae_tile
        self.denoise = denoise
        self.thumbnail = thumbnail

    def do_cfg(self):
        return self.cfg > 1.0

    def get_valid_pipeline(self, group: str, pipeline: Optional[str] = None) -> str:
        pipeline = pipeline or self.pipeline

        # if the correct pipeline was already requested, simply use that
        if group == pipeline:
            return pipeline

        # otherwise, check for additional allowed pipelines
        if group == "img2img":
            if pipeline in [
                "controlnet",
                "img2img-sdxl",
                "lpw",
                "panorama",
                "panorama-sdxl",
                "pix2pix",
            ]:
                return pipeline
            elif pipeline == "txt2img-sdxl":
                return "img2img-sdxl"
        elif group == "inpaint":
            if pipeline in ["controlnet", "lpw", "panorama"]:
                return pipeline
        elif group == "txt2img":
            if pipeline in ["lpw", "panorama", "panorama-sdxl", "txt2img-sdxl"]:
                return pipeline

        logger.debug("pipeline %s is not valid for %s", pipeline, group)
        return group

    def is_control(self):
        return self.pipeline in ["controlnet", "controlnet-sdxl"]

    def is_lpw(self):
        return self.pipeline == "lpw"

    def is_panorama(self):
        return self.pipeline in ["panorama", "panorama-sdxl"]

    def is_pix2pix(self):
        return self.pipeline == "pix2pix"

    def is_xl(self):
        return self.pipeline.endswith("-sdxl")

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
            "tiled_vae": self.tiled_vae,
            "unet_overlap": self.unet_overlap,
            "unet_tile": self.unet_tile,
            "vae_overlap": self.vae_overlap,
            "vae_tile": self.vae_tile,
            "denoise": self.denoise,
            "thumbnail": self.thumbnail,
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
            kwargs.get("tiled_vae", self.tiled_vae),
            kwargs.get("unet_overlap", self.unet_overlap),
            kwargs.get("unet_tile", self.unet_tile),
            kwargs.get("vae_overlap", self.vae_overlap),
            kwargs.get("vae_tile", self.vae_tile),
            kwargs.get("denoise", self.denoise),
            kwargs.get("thumbnail", self.thumbnail),
        )


class StageParams:
    """
    Parameters for a chained pipeline stage
    """

    def __init__(
        self,
        name: Optional[str] = None,
        outscale: int = 1,
        tile_order: str = TileOrder.spiral,
        tile_size: int = SizeChart.auto,
        # batch_size: int = 1,
    ) -> None:
        self.name = name
        self.outscale = outscale
        self.tile_order = tile_order
        self.tile_size = tile_size

    def with_args(
        self,
        name: Optional[str] = None,
        outscale: Optional[int] = None,
        tile_order: Optional[str] = None,
        tile_size: Optional[int] = None,
        **kwargs,
    ):
        logger.debug("ignoring extra kwargs for stage: %s", kwargs)
        return StageParams(
            name=coalesce(name, self.name),
            outscale=coalesce(outscale, self.outscale),
            tile_order=coalesce(tile_order, self.tile_order),
            tile_size=coalesce(tile_size, self.tile_size),
        )


UpscaleOrder = Literal["correction-first", "correction-last", "correction-both"]


class UpscaleParams:
    def __init__(
        self,
        upscale_model: str,
        correction_model: Optional[str] = None,
        denoise: float = 0.5,
        upscale=True,
        faces=True,
        face_outscale: int = 1,
        face_strength: float = 0.5,
        outscale: int = 1,
        scale: int = 4,
        pre_pad: int = 0,
        tile_pad: int = 10,
        upscale_order: UpscaleOrder = "correction-first",
    ) -> None:
        self.upscale_model = upscale_model
        self.correction_model = correction_model
        self.denoise = denoise
        self.upscale = upscale
        self.faces = faces
        self.face_outscale = face_outscale
        self.face_strength = face_strength
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
            upscale=self.upscale,
            faces=self.faces,
            face_outscale=self.face_outscale,
            face_strength=self.face_strength,
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
            "upscale": self.upscale,
            "faces": self.faces,
            "face_outscale": self.face_outscale,
            "face_strength": self.face_strength,
            "outscale": self.outscale,
            "pre_pad": self.pre_pad,
            "scale": self.scale,
            "tile_pad": self.tile_pad,
            "upscale_order": self.upscale_order,
        }

    def with_args(
        self,
        upscale_model: Optional[str] = None,
        correction_model: Optional[str] = None,
        denoise: Optional[float] = None,
        upscale: Optional[bool] = None,
        faces: Optional[bool] = None,
        face_outscale: Optional[int] = None,
        face_strength: Optional[float] = None,
        outscale: Optional[int] = None,
        scale: Optional[int] = None,
        pre_pad: Optional[int] = None,
        tile_pad: Optional[int] = None,
        upscale_order: Optional[UpscaleOrder] = None,
        **kwargs,
    ):
        logger.debug("ignoring extra kwargs for upscale: %s", kwargs)
        return UpscaleParams(
            upscale_model=coalesce(upscale_model, self.upscale_model),
            correction_model=coalesce(correction_model, self.correction_model),
            denoise=coalesce(denoise, self.denoise),
            upscale=coalesce(upscale, self.upscale),
            faces=coalesce(faces, self.faces),
            face_outscale=coalesce(face_outscale, self.face_outscale),
            face_strength=coalesce(face_strength, self.face_strength),
            outscale=coalesce(outscale, self.outscale),
            scale=coalesce(scale, self.scale),
            pre_pad=coalesce(pre_pad, self.pre_pad),
            tile_pad=coalesce(tile_pad, self.tile_pad),
            upscale_order=coalesce(upscale_order, self.upscale_order),
        )


UpscaleMethod = Literal["bilinear", "lanczos", "upscale"]


class HighresParams:
    def __init__(
        self,
        enabled: bool,
        scale: int,
        steps: int,
        strength: float,
        method: UpscaleMethod = "lanczos",
        iterations: int = 1,
    ):
        self.enabled = enabled
        self.scale = scale
        self.steps = steps
        self.strength = strength
        self.method = method
        self.iterations = iterations

    def outscale(self) -> int:
        return self.scale**self.iterations

    def resize(self, size: Size) -> Size:
        outscale = self.outscale()
        return Size(
            size.width * outscale,
            size.height * outscale,
        )

    def tojson(self):
        return {
            "enabled": self.enabled,
            "iterations": self.iterations,
            "method": self.method,
            "scale": self.scale,
            "steps": self.steps,
            "strength": self.strength,
        }

    def with_args(
        self,
        enabled: Optional[bool] = None,
        scale: Optional[int] = None,
        steps: Optional[int] = None,
        strength: Optional[float] = None,
        method: Optional[UpscaleMethod] = None,
        iterations: Optional[int] = None,
        **kwargs,
    ):
        logger.debug("ignoring extra kwargs for highres: %s", kwargs)
        return HighresParams(
            enabled=coalesce(enabled, self.enabled),
            scale=coalesce(scale, self.scale),
            steps=coalesce(steps, self.steps),
            strength=coalesce(strength, self.strength),
            method=coalesce(method, self.method),
            iterations=coalesce(iterations, self.iterations),
        )
