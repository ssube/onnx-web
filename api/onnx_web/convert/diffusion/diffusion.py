###
# Parts of this file are copied or derived from:
#   https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
#
# Originally by https://github.com/huggingface
# Those portions *are not* covered by the MIT licensed used for the rest of the onnx-web project.
# ...diffusers.pipelines.pipeline_onnx_stable_diffusion_upscale
# HuggingFace code used under the Apache License, Version 2.0
#   https://github.com/huggingface/diffusers/blob/main/LICENSE
###

from logging import getLogger
from os import makedirs, path
from pathlib import Path
from shutil import rmtree
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers import (
    AutoencoderKL,
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    StableDiffusionUpscalePipeline,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from onnx import load_model, save_model

from ...constants import ONNX_MODEL, ONNX_WEIGHTS
from ...diffusers.load import optimize_pipeline
from ...diffusers.pipelines.controlnet import OnnxStableDiffusionControlNetPipeline
from ...diffusers.pipelines.upscale import OnnxStableDiffusionUpscalePipeline
from ...diffusers.version_safe_diffusers import AttnProcessor
from ...models.cnet import UNet2DConditionModel_CNet
from ...utils import run_gc
from ..client import fetch_model
from ..client.huggingface import HuggingfaceClient
from ..utils import (
    RESOLVE_FORMATS,
    ConversionContext,
    check_ext,
    is_torch_2_0,
    load_tensor,
    onnx_export,
    remove_prefix,
)
from .checkpoint import convert_extract_checkpoint

logger = getLogger(__name__)

CONVERT_PIPELINES = {
    "controlnet": OnnxStableDiffusionControlNetPipeline,
    "img2img": StableDiffusionPipeline,
    "inpaint": StableDiffusionPipeline,
    "lpw": StableDiffusionPipeline,
    "panorama": StableDiffusionPipeline,
    "pix2pix": StableDiffusionInstructPix2PixPipeline,
    "txt2img": StableDiffusionPipeline,
    "upscale": StableDiffusionUpscalePipeline,
}


def get_model_version(
    source,
    map_location,
    size=None,
    version=None,
) -> Tuple[bool, Dict[str, Union[bool, int, str]]]:
    v2 = version is not None and "v2" in version
    opts = {
        "extract_ema": True,
    }

    try:
        checkpoint = load_tensor(source, map_location=map_location)

        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        else:
            logger.trace("global_step key not found in model")
            global_step = None

        if size is None:
            # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
            # as it relies on a brittle global step parameter here
            size = 512 if global_step == 875000 else 768

        opts["image_size"] = size

        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
            v2 = True
            if size != 512:
                # v2.1 needs to upcast attention
                logger.trace("setting upcast_attention")
                opts["upcast_attention"] = True

        if v2 and size != 512:
            opts["model_type"] = "FrozenOpenCLIPEmbedder"
            opts["prediction_type"] = "v_prediction"
        else:
            opts["model_type"] = "FrozenCLIPEmbedder"
            opts["prediction_type"] = "epsilon"
    except Exception:
        logger.debug("unable to load tensor for version check")

    return (v2, opts)


@torch.no_grad()
def convert_diffusion_diffusers_cnet(
    conversion: ConversionContext,
    name: str,
    source: str,
    device: str,
    output_path: Path,
    dtype,
    unet_in_channels,
    unet_sample_size,
    num_tokens,
    text_hidden_size,
    unet: Optional[Any] = None,
    v2: Optional[bool] = False,
):
    if unet is not None:
        logger.debug("creating CNet from existing UNet config")
        cnet_tmp = path.join(conversion.cache_path, f"{name}-cnet")
        makedirs(cnet_tmp, exist_ok=True)

        unet.save_pretrained(cnet_tmp)
        pipe_cnet = UNet2DConditionModel_CNet.from_pretrained(
            cnet_tmp, low_cpu_mem_usage=False
        )
    else:
        logger.debug("loading CNet from pretrained UNet config")
        pipe_cnet = UNet2DConditionModel_CNet.from_pretrained(
            source, subfolder="unet", low_cpu_mem_usage=False
        )

    pipe_cnet = pipe_cnet.to(device=device, dtype=dtype)
    run_gc()

    if is_torch_2_0:
        pipe_cnet.set_attn_processor(AttnProcessor())

    optimize_pipeline(conversion, pipe_cnet)

    cnet_path = output_path / "cnet" / ONNX_MODEL
    onnx_export(
        pipe_cnet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(
                device=device, dtype=dtype
            ),
            torch.randn(2).to(device=device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 320, unet_sample_size, unet_sample_size).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 320, unet_sample_size // 2, unet_sample_size // 2).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 640, unet_sample_size // 2, unet_sample_size // 2).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 640, unet_sample_size // 2, unet_sample_size // 2).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 640, unet_sample_size // 4, unet_sample_size // 4).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 4, unet_sample_size // 4).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 4, unet_sample_size // 4).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 8, unet_sample_size // 8).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 8, unet_sample_size // 8).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 8, unet_sample_size // 8).to(
                device=device, dtype=dtype
            ),
            torch.randn(2, 1280, unet_sample_size // 8, unet_sample_size // 8).to(
                device=device, dtype=dtype
            ),
            False,
        ),
        ordered_input_names=[
            "sample",
            "timestep",
            "encoder_hidden_states",
            "down_block_0",
            "down_block_1",
            "down_block_2",
            "down_block_3",
            "down_block_4",
            "down_block_5",
            "down_block_6",
            "down_block_7",
            "down_block_8",
            "down_block_9",
            "down_block_10",
            "down_block_11",
            "mid_block_additional_residual",
            "return_dict",
        ],
        output_names=[
            "out_sample"
        ],  # has to be different from "sample" for correct tracing
        dynamic_axes={
            "sample": {
                0: "cnet_sample_batch",
                1: "cnet_sample_channels",
                2: "cnet_sample_height",
                3: "cnet_sample_width",
            },
            "timestep": {0: "cnet_timestep_batch"},
            "encoder_hidden_states": {0: "cnet_ehs_batch", 1: "cnet_ehs_sequence"},
            "down_block_0": {
                0: "cnet_db0_batch",
                2: "cnet_db0_height",
                3: "cnet_db0_width",
            },
            "down_block_1": {
                0: "cnet_db1_batch",
                2: "cnet_db1_height",
                3: "cnet_db1_width",
            },
            "down_block_2": {
                0: "cnet_db2_batch",
                2: "cnet_db2_height",
                3: "cnet_db2_width",
            },
            "down_block_3": {
                0: "cnet_db3_batch",
                2: "cnet_db3_height2",
                3: "cnet_db3_width2",
            },
            "down_block_4": {
                0: "cnet_db4_batch",
                2: "cnet_db4_height2",
                3: "cnet_db4_width2",
            },
            "down_block_5": {
                0: "cnet_db5_batch",
                2: "cnet_db5_height2",
                3: "cnet_db5_width2",
            },
            "down_block_6": {
                0: "cnet_db6_batch",
                2: "cnet_db6_height4",
                3: "cnet_db6_width4",
            },
            "down_block_7": {
                0: "cnet_db7_batch",
                2: "cnet_db7_height4",
                3: "cnet_db7_width4",
            },
            "down_block_8": {
                0: "cnet_db8_batch",
                2: "cnet_db8_height4",
                3: "cnet_db8_width4",
            },
            "down_block_9": {
                0: "cnet_db9_batch",
                2: "cnet_db9_height8",
                3: "cnet_db9_width8",
            },
            "down_block_10": {
                0: "cnet_db10_batch",
                2: "cnet_db10_height8",
                3: "cnet_db10_width8",
            },
            "down_block_11": {
                0: "cnet_db11_batch",
                2: "cnet_db11_height8",
                3: "cnet_db11_width8",
            },
            "mid_block_additional_residual": {
                0: "cnet_mbar_batch",
                2: "cnet_mbar_height8",
                3: "cnet_mbar_width8",
            },
        },
        output_path=cnet_path,
        opset=conversion.opset,
        half=conversion.half,
        external_data=True,  # UNet is > 2GB, so the weights need to be split
        v2=v2,
    )
    del pipe_cnet
    run_gc()

    return cnet_path


@torch.no_grad()
def collate_cnet(cnet_path):
    logger.debug("collating CNet external tensors")
    cnet_model_path = str(cnet_path.absolute().as_posix())
    cnet_dir = path.dirname(cnet_model_path)
    cnet = load_model(cnet_model_path)

    # clean up existing tensor files
    rmtree(cnet_dir)
    makedirs(cnet_dir)

    # collate external tensor files into one
    save_model(
        cnet,
        cnet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ONNX_WEIGHTS,
        convert_attribute=False,
    )
    del cnet
    run_gc()


@torch.no_grad()
def convert_diffusion_diffusers(
    conversion: ConversionContext,
    model: Dict,
    format: Optional[str],
) -> Tuple[bool, str]:
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    name = str(model.get("name")).strip()
    source = model.get("source")

    # optional
    config = model.get("config", None)
    image_size = model.get("image_size", None)
    pipe_type = model.get("pipeline", "txt2img")
    single_vae = model.get("single_vae", False)
    replace_vae = model.get("vae", None)
    version = model.get("version", None)

    device = conversion.training_device
    dtype = conversion.torch_dtype()
    logger.debug("using Torch dtype %s for pipeline", dtype)

    config_path = (
        None if config is None else path.join(conversion.model_path, "config", config)
    )
    dest_path = path.join(conversion.model_path, name)
    model_index = path.join(dest_path, "model_index.json")
    model_cnet = path.join(dest_path, "cnet", ONNX_MODEL)
    model_hash = path.join(dest_path, "hash.txt")

    # diffusers go into a directory rather than .onnx file
    logger.info(
        "converting Stable Diffusion model %s: %s -> %s/", name, source, dest_path
    )

    if single_vae:
        logger.info("converting model with single VAE")

    cnet_only = False
    if path.exists(dest_path) and path.exists(model_index):
        if "hash" in model and not path.exists(model_hash):
            logger.info("ONNX model does not have hash file, adding one")
            with open(model_hash, "w") as f:
                f.write(model["hash"])

        if not single_vae and not path.exists(model_cnet):
            logger.info(
                "ONNX model was converted without a ControlNet UNet, converting one"
            )
            cnet_only = True
        else:
            logger.info("ONNX model already exists, skipping")
            return (False, dest_path)

    cache_path = fetch_model(conversion, name, source, format=format)

    pipe_class = CONVERT_PIPELINES.get(pipe_type)
    v2, pipe_args = get_model_version(
        cache_path, conversion.map_location, size=image_size, version=version
    )

    is_inpainting = False
    if pipe_type == "inpaint":
        pipe_args["num_in_channels"] = 9
        is_inpainting = True

    if format == "safetensors":
        pipe_args["from_safetensors"] = True

    torch_source = None
    if path.exists(cache_path):
        if path.isdir(cache_path):
            logger.debug("loading pipeline from diffusers directory: %s", source)
            pipeline = pipe_class.from_pretrained(
                cache_path,
                torch_dtype=dtype,
                use_auth_token=conversion.token,
            ).to(device)
        else:
            if conversion.extract:
                logger.debug("extracting SD checkpoint to Torch models: %s", source)
                torch_source = convert_extract_checkpoint(
                    conversion,
                    cache_path,
                    f"{name}-torch",
                    is_inpainting=is_inpainting,
                    config_file=config,
                    vae_file=replace_vae,
                )
                logger.debug(
                    "loading pipeline from extracted checkpoint: %s", torch_source
                )
                pipeline = pipe_class.from_pretrained(
                    torch_source,
                    torch_dtype=dtype,
                ).to(device)

                # VAE replacement already happened during extraction, skip
                replace_vae = None
            else:
                logger.debug("loading pipeline from SD checkpoint: %s", source)
                pipeline = download_from_original_stable_diffusion_ckpt(
                    cache_path,
                    original_config_file=config_path,
                    pipeline_class=pipe_class,
                    **pipe_args,
                ).to(device, torch_dtype=dtype)
    elif source.startswith(HuggingfaceClient.protocol):
        hf_path = remove_prefix(source, HuggingfaceClient.protocol)
        logger.debug("downloading pretrained model from Huggingface hub: %s", hf_path)
        pipeline = pipe_class.from_pretrained(
            hf_path,
            torch_dtype=dtype,
            use_auth_token=conversion.token,
        ).to(device)
    else:
        logger.warning(
            "pipeline source not found and protocol not recognized: %s", source
        )
        raise ValueError(
            f"pipeline source not found and protocol not recognized: {source}"
        )

    if replace_vae is not None:
        vae_path = path.join(conversion.model_path, replace_vae)
        vae_file = check_ext(vae_path, RESOLVE_FORMATS)
        if vae_file[0]:
            pipeline.vae = AutoencoderKL.from_single_file(vae_path)
        else:
            pipeline.vae = AutoencoderKL.from_pretrained(replace_vae)

    if is_torch_2_0:
        pipeline.unet.set_attn_processor(AttnProcessor())
        pipeline.vae.set_attn_processor(AttnProcessor())

    optimize_pipeline(conversion, pipeline)

    output_path = Path(dest_path)

    # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    if not cnet_only:
        encoder_path = output_path / "text_encoder" / ONNX_MODEL
        logger.info("exporting text encoder to %s", encoder_path)
        onnx_export(
            pipeline.text_encoder,
            # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
            model_args=(
                text_input.input_ids.to(device=device, dtype=torch.int32),
                None,  # attention mask
                None,  # position ids
                None,  # output attentions
                torch.tensor(True).to(device=device, dtype=torch.bool),
            ),
            output_path=encoder_path,
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output", "hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=conversion.opset,
            half=conversion.half,
        )

    del pipeline.text_encoder
    run_gc()

    # UNET
    logger.debug("UNET config: %s", pipeline.unet.config)
    if single_vae:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "class_labels"]
        unet_scale = torch.tensor(4).to(device=device, dtype=torch.long)
    else:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "return_dict"]
        unet_scale = torch.tensor(False).to(device=device, dtype=torch.bool)

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / ONNX_MODEL

    if not cnet_only:
        logger.info("exporting UNet to %s", unet_path)
        onnx_export(
            pipeline.unet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(
                    device=device, dtype=dtype
                ),
                torch.randn(2).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(
                    device=device, dtype=dtype
                ),
                unet_scale,
            ),
            output_path=unet_path,
            ordered_input_names=unet_inputs,
            # has to be different from "sample" for correct tracing
            output_names=["out_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "timestep": {0: "batch"},
                "encoder_hidden_states": {0: "batch", 1: "sequence"},
            },
            opset=conversion.opset,
            half=conversion.half,
            external_data=True,
            v2=v2,
        )

    cnet_path = None
    if conversion.control and not single_vae and conversion.share_unet:
        logger.debug("converting CNet from loaded UNet")
        cnet_path = convert_diffusion_diffusers_cnet(
            conversion,
            name,
            source,
            device,
            output_path,
            dtype,
            unet_in_channels,
            unet_sample_size,
            num_tokens,
            text_hidden_size,
            unet=pipeline.unet,
            v2=v2,
        )

    del pipeline.unet
    run_gc()

    if conversion.control and not single_vae and not conversion.share_unet:
        cnet_source = torch_source or cache_path
        logger.info("loading and converting CNet from %s", cnet_source)
        cnet_path = convert_diffusion_diffusers_cnet(
            conversion,
            name,
            cnet_source,
            device,
            output_path,
            dtype,
            unet_in_channels,
            unet_sample_size,
            num_tokens,
            text_hidden_size,
            unet=None,
            v2=v2,
        )

    if cnet_path is not None:
        collate_cnet(cnet_path)

    if cnet_only:
        logger.info("done converting CNet")
        return (True, dest_path)

    logger.debug("collating UNet external tensors")
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = path.dirname(unet_model_path)
    unet = load_model(unet_model_path)

    # clean up existing tensor files
    rmtree(unet_dir)
    makedirs(unet_dir)

    # collate external tensor files into one
    save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ONNX_WEIGHTS,
        convert_attribute=False,
    )

    del unet
    run_gc()

    if single_vae:
        logger.debug("VAE config: %s", pipeline.vae.config)

        # SINGLE VAE
        vae_only = pipeline.vae
        vae_latent_channels = vae_only.config.latent_channels
        # forward only through the decoder part
        vae_only.forward = vae_only.decode

        vae_path = output_path / "vae" / ONNX_MODEL
        logger.info("exporting VAE to %s", vae_path)
        onnx_export(
            vae_only,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=device, dtype=dtype),
                False,
            ),
            output_path=vae_path,
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=conversion.opset,
            half=conversion.half,
        )
    else:
        # VAE ENCODER
        vae_encoder = pipeline.vae
        vae_in_channels = vae_encoder.config.in_channels
        vae_sample_size = vae_encoder.config.sample_size
        # need to get the raw tensor output (sample) from the encoder
        vae_encoder.forward = lambda sample, return_dict: vae_encoder.encode(
            sample, return_dict
        )[0].sample()

        vae_path = output_path / "vae_encoder" / ONNX_MODEL
        logger.info("exporting VAE encoder to %s", vae_path)
        onnx_export(
            vae_encoder,
            model_args=(
                torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(
                    device=device, dtype=dtype
                ),
                False,
            ),
            output_path=vae_path,
            ordered_input_names=["sample", "return_dict"],
            output_names=["latent_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=conversion.opset,
            half=False,  # https://github.com/ssube/onnx-web/issues/290
        )

        # VAE DECODER
        vae_decoder = pipeline.vae
        vae_latent_channels = vae_decoder.config.latent_channels
        # forward only through the decoder part
        vae_decoder.forward = vae_encoder.decode

        vae_path = output_path / "vae_decoder" / ONNX_MODEL
        logger.info("exporting VAE decoder to %s", vae_path)
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=device, dtype=dtype),
                False,
            ),
            output_path=vae_path,
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=conversion.opset,
            half=conversion.half,
        )

    del pipeline.vae
    run_gc()

    if single_vae:
        logger.debug("reloading diffusion model with upscaling pipeline")
        onnx_pipeline = OnnxStableDiffusionUpscalePipeline(
            vae=OnnxRuntimeModel.from_pretrained(output_path / "vae"),
            text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
            tokenizer=pipeline.tokenizer,
            low_res_scheduler=pipeline.scheduler,
            unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
            scheduler=pipeline.scheduler,
        )
    else:
        logger.debug("reloading diffusion model with default pipeline")
        onnx_pipeline = OnnxStableDiffusionPipeline(
            vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
            vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
            text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
            tokenizer=pipeline.tokenizer,
            unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
            scheduler=pipeline.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )

    logger.info("exporting pretrained ONNX model to %s", output_path)

    onnx_pipeline.save_pretrained(output_path)
    logger.info("ONNX pipeline saved to %s", output_path)

    del pipeline
    del onnx_pipeline
    run_gc()

    if conversion.reload:
        if single_vae:
            _ = OnnxStableDiffusionUpscalePipeline.from_pretrained(
                output_path, provider="CPUExecutionProvider"
            )
        else:
            _ = OnnxStableDiffusionPipeline.from_pretrained(
                output_path, provider="CPUExecutionProvider"
            )

        logger.info("ONNX pipeline is loadable")
    else:
        logger.debug("skipping ONNX reload test")

    return (True, dest_path)
