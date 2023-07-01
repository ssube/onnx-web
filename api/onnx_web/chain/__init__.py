from .base import ChainPipeline, PipelineStage, StageCallback, StageParams
from .blend_img2img import BlendImg2ImgStage
from .blend_inpaint import BlendInpaintStage
from .blend_linear import BlendLinearStage
from .blend_mask import BlendMaskStage
from .correct_codeformer import CorrectCodeformerStage
from .correct_gfpgan import CorrectGFPGANStage
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
from .upscale_stable_diffusion import UpscaleStableDiffusionStage
from .upscale_swinir import UpscaleSwinIRStage

CHAIN_STAGES = {
    "blend-img2img": BlendImg2ImgStage,
    "blend-inpaint": BlendInpaintStage,
    "blend-linear": BlendLinearStage,
    "blend-mask": BlendMaskStage,
    "correct-codeformer": CorrectCodeformerStage,
    "correct-gfpgan": CorrectGFPGANStage,
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
    "upscale-stable-diffusion": UpscaleStableDiffusionStage,
    "upscale-swinir": UpscaleSwinIRStage,
}
