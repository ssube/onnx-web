from logging import getLogger

from .edit_safety import EditSafetyStage
from .edit_text import EditTextStage

from .base import BaseStage
from .blend_denoise_fastnlmeans import BlendDenoiseFastNLMeansStage
from .blend_denoise_localstd import BlendDenoiseLocalStdStage
from .blend_grid import BlendGridStage
from .blend_img2img import BlendImg2ImgStage
from .blend_linear import BlendLinearStage
from .blend_mask import BlendMaskStage
from .correct_codeformer import CorrectCodeformerStage
from .correct_gfpgan import CorrectGFPGANStage
from .edit_metadata import EditMetadataStage
from .persist_disk import PersistDiskStage
from .persist_s3 import PersistS3Stage
from .reduce_crop import ReduceCropStage
from .reduce_thumbnail import ReduceThumbnailStage
from .source_noise import SourceNoiseStage
from .source_s3 import SourceS3Stage
from .source_txt2img import SourceTxt2ImgStage
from .source_url import SourceURLStage
from .upscale_bsrgan import UpscaleBSRGANStage
from .upscale_highres import UpscaleHighresStage
from .upscale_outpaint import UpscaleOutpaintStage
from .upscale_resrgan import UpscaleRealESRGANStage
from .upscale_simple import UpscaleSimpleStage
from .upscale_stable_diffusion import UpscaleStableDiffusionStage
from .upscale_swinir import UpscaleSwinIRStage

logger = getLogger(__name__)

CHAIN_STAGES = {
    "blend-denoise": BlendDenoiseFastNLMeansStage,
    "blend-denoise-fastnlmeans": BlendDenoiseFastNLMeansStage,
    "blend-denoise-localstd": BlendDenoiseLocalStdStage,
    "blend-img2img": BlendImg2ImgStage,
    "blend-inpaint": UpscaleOutpaintStage,
    "blend-grid": BlendGridStage,
    "blend-linear": BlendLinearStage,
    "blend-mask": BlendMaskStage,
    "correct-codeformer": CorrectCodeformerStage,
    "correct-gfpgan": CorrectGFPGANStage,
    "edit-metadata": EditMetadataStage,
    "edit-safety": EditSafetyStage,
    "edit-text": EditTextStage,
    "persist-disk": PersistDiskStage,
    "persist-s3": PersistS3Stage,
    "reduce-crop": ReduceCropStage,
    "reduce-thumbnail": ReduceThumbnailStage,
    "source-noise": SourceNoiseStage,
    "source-s3": SourceS3Stage,
    "source-txt2img": SourceTxt2ImgStage,
    "source-url": SourceURLStage,
    "upscale-bsrgan": UpscaleBSRGANStage,
    "upscale-highres": UpscaleHighresStage,
    "upscale-outpaint": UpscaleOutpaintStage,
    "upscale-resrgan": UpscaleRealESRGANStage,
    "upscale-simple": UpscaleSimpleStage,
    "upscale-stable-diffusion": UpscaleStableDiffusionStage,
    "upscale-swinir": UpscaleSwinIRStage,
}


def add_stage(name: str, stage: BaseStage) -> bool:
    global CHAIN_STAGES

    if name in CHAIN_STAGES:
        logger.warning("cannot replace stage: %s", name)
        return False
    else:
        CHAIN_STAGES[name] = stage
        return True
