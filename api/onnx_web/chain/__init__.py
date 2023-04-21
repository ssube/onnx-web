from .base import ChainPipeline, PipelineStage, StageCallback, StageParams
from .blend_controlnet import blend_controlnet
from .blend_img2img import blend_img2img
from .blend_inpaint import blend_inpaint
from .blend_linear import blend_linear
from .blend_mask import blend_mask
from .blend_pix2pix import blend_pix2pix
from .correct_codeformer import correct_codeformer
from .correct_gfpgan import correct_gfpgan
from .persist_disk import persist_disk
from .persist_s3 import persist_s3
from .reduce_crop import reduce_crop
from .reduce_thumbnail import reduce_thumbnail
from .source_noise import source_noise
from .source_s3 import source_s3
from .source_txt2img import source_txt2img
from .source_url import source_url
from .upscale_bsrgan import upscale_bsrgan
from .upscale_outpaint import upscale_outpaint
from .upscale_resrgan import upscale_resrgan
from .upscale_stable_diffusion import upscale_stable_diffusion
from .upscale_swinir import upscale_swinir

CHAIN_STAGES = {
    "blend-controlnet": blend_controlnet,
    "blend-img2img": blend_img2img,
    "blend-inpaint": blend_inpaint,
    "blend-linear": blend_linear,
    "blend-mask": blend_mask,
    "blend-pix2pix": blend_pix2pix,
    "correct-codeformer": correct_codeformer,
    "correct-gfpgan": correct_gfpgan,
    "persist-disk": persist_disk,
    "persist-s3": persist_s3,
    "reduce-crop": reduce_crop,
    "reduce-thumbnail": reduce_thumbnail,
    "source-noise": source_noise,
    "source-s3": source_s3,
    "source-txt2img": source_txt2img,
    "source-url": source_url,
    "upscale-bsrgan": upscale_bsrgan,
    "upscale-outpaint": upscale_outpaint,
    "upscale-resrgan": upscale_resrgan,
    "upscale-stable-diffusion": upscale_stable_diffusion,
    "upscale-swinir": upscale_swinir,
}
