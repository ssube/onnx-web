from enum import IntEnum
from typing import Any, Dict, Literal, Optional, Tuple, Union


class SizeChart(IntEnum):
    mini = 128      # small tile for very expensive models
    half = 256      # half tile for outpainting
    auto = 512      # auto tile size
    hd1k = 2**10
    hd2k = 2**11
    hd4k = 2**12
    hd8k = 2**13
    hd16k = 2**14
    hd64k = 2**16


Param = Union[str, int, float]
Point = Tuple[int, int]


class Border:
    def __init__(self, left: int, right: int, top: int, bottom: int) -> None:
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def __str__(self) -> str:
        return '%s %s %s %s' % (self.left, self.top, self.right, self.bottom)

    @classmethod
    def even(cls, all: int):
        return Border(all, all, all, all)


class Size:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def __str__(self) -> str:
        return '%sx%s' % (self.width, self.height)

    def add_border(self, border: Border):
        return Size(border.left + self.width + border.right, border.top + self.height + border.right)

    def tojson(self) -> Dict[str, int]:
        return {
            'height': self.height,
            'width': self.width,
        }


class ImageParams:
    def __init__(
        self,
        model: str,
        provider: str,
        scheduler: Any,
        prompt: str,
        negative_prompt: Optional[str],
        cfg: float,
        steps: int,
        seed: int
    ) -> None:
        self.model = model
        self.provider = provider
        self.scheduler = scheduler
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.cfg = cfg
        self.steps = steps
        self.seed = seed

    def tojson(self) -> Dict[str, Param]:
        return {
            'model': self.model,
            'provider': self.provider,
            'scheduler': self.scheduler.__name__,
            'seed': self.seed,
            'prompt': self.prompt,
            'cfg': self.cfg,
            'negativePrompt': self.negative_prompt,
            'steps': self.steps,
        }


class StageParams:
    '''
    Parameters for a chained pipeline stage
    '''

    def __init__(
        self,
        name: Optional[str] = None,
        tile_size: int = SizeChart.auto,
        outscale: int = 1,
        # batch_size: int = 1,
    ) -> None:
        self.name = name
        self.tile_size = tile_size
        self.outscale = outscale


class UpscaleParams():
    def __init__(
        self,
        upscale_model: str,
        provider: str,
        correction_model: Optional[str] = None,
        denoise: float = 0.5,
        faces=True,
        face_strength: float = 0.5,
        format: Literal['onnx', 'pth'] = 'onnx',
        half=False,
        outscale: int = 1,
        scale: int = 4,
        pre_pad: int = 0,
        tile_pad: int = 10,
    ) -> None:
        self.upscale_model = upscale_model
        self.provider = provider
        self.correction_model = correction_model
        self.denoise = denoise
        self.faces = faces
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
            self.provider,
            correction_model=self.correction_model,
            denoise=self.denoise,
            faces=self.faces,
            face_strength=self.face_strength,
            format=self.format,
            half=self.half,
            outscale=scale,
            scale=scale,
            pre_pad=self.pre_pad,
            tile_pad=self.tile_pad,
        )

    def resize(self, size: Size) -> Size:
        return Size(size.width * self.outscale, size.height * self.outscale)
