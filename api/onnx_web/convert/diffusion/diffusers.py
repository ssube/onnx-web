###
# Parts of this file are copied or derived from:
#   https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
#
# Originally by https://github.com/huggingface
# Those portions *are not* covered by the MIT licensed used for the rest of the onnx-web project.
#
# HuggingFace code used under the Apache License, Version 2.0
#   https://github.com/huggingface/diffusers/blob/main/LICENSE
###

from logging import getLogger
from os import mkdir, path
from pathlib import Path
from shutil import rmtree
from typing import Dict

import torch
from diffusers import (
    AutoencoderKL,
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    StableDiffusionPipeline,
)
from onnx import load, save_model
from torch.onnx import export

from ...diffusion.load import optimize_pipeline
from ...diffusion.pipeline_onnx_stable_diffusion_upscale import (
    OnnxStableDiffusionUpscalePipeline,
)
from ..utils import ConversionContext

logger = getLogger(__name__)


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export(
        model,
        model_args,
        f=output_path.as_posix(),
        input_names=ordered_input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=opset,
    )


@torch.no_grad()
def convert_diffusion_diffusers(
    ctx: ConversionContext,
    model: Dict,
    source: str,
):
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    name = model.get("name")
    source = source or model.get("source")
    single_vae = model.get("single_vae")
    replace_vae = model.get("vae")

    dtype = torch.float16 if ctx.half else torch.float32
    dest_path = path.join(ctx.model_path, name)

    # diffusers go into a directory rather than .onnx file
    logger.info(
        "converting Stable Diffusion model %s: %s -> %s/", name, source, dest_path
    )

    if single_vae:
        logger.info("converting model with single VAE")

    if path.exists(dest_path):
        logger.info("ONNX model already exists, skipping")
        return

    pipeline = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=dtype,
        use_auth_token=ctx.token,
    ).to(ctx.training_device)
    output_path = Path(dest_path)

    optimize_pipeline(ctx, pipeline)

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
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(
            text_input.input_ids.to(device=ctx.training_device, dtype=torch.int32)
        ),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        opset=ctx.opset,
    )
    del pipeline.text_encoder

    logger.debug("UNET config: %s", pipeline.unet.config)

    # UNET
    if single_vae:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "class_labels"]
        unet_scale = torch.tensor(4).to(device=ctx.training_device, dtype=torch.long)
    else:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "return_dict"]
        unet_scale = torch.tensor(False).to(
            device=ctx.training_device, dtype=torch.bool
        )

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / "model.onnx"
    onnx_export(
        pipeline.unet,
        model_args=(
            torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(
                device=ctx.training_device, dtype=dtype
            ),
            torch.randn(2).to(device=ctx.training_device, dtype=dtype),
            torch.randn(2, num_tokens, text_hidden_size).to(
                device=ctx.training_device, dtype=dtype
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
        opset=ctx.opset,
        use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
    )
    unet_model_path = str(unet_path.absolute().as_posix())
    unet_dir = path.dirname(unet_model_path)
    unet = load(unet_model_path)
    # clean up existing tensor files
    rmtree(unet_dir)
    mkdir(unet_dir)
    # collate external tensor files into one
    save_model(
        unet,
        unet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.pb",
        convert_attribute=False,
    )
    del pipeline.unet

    if replace_vae is not None:
        logger.debug("loading custom VAE: %s", replace_vae)
        vae = AutoencoderKL.from_pretrained(replace_vae)
        pipeline.vae = vae

    if single_vae:
        logger.debug("VAE config: %s", pipeline.vae.config)

        # SINGLE VAE
        vae_only = pipeline.vae
        vae_latent_channels = vae_only.config.latent_channels
        vae_out_channels = vae_only.config.out_channels
        # forward only through the decoder part
        vae_only.forward = vae_only.decode
        onnx_export(
            vae_only,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=ctx.training_device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae" / "model.onnx",
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=ctx.opset,
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
        onnx_export(
            vae_encoder,
            model_args=(
                torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(
                    device=ctx.training_device, dtype=dtype
                ),
                False,
            ),
            output_path=output_path / "vae_encoder" / "model.onnx",
            ordered_input_names=["sample", "return_dict"],
            output_names=["latent_sample"],
            dynamic_axes={
                "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=ctx.opset,
        )

        # VAE DECODER
        vae_decoder = pipeline.vae
        vae_latent_channels = vae_decoder.config.latent_channels
        vae_out_channels = vae_decoder.config.out_channels
        # forward only through the decoder part
        vae_decoder.forward = vae_encoder.decode
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=ctx.training_device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_decoder" / "model.onnx",
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=ctx.opset,
        )

    del pipeline.vae

    # SAFETY CHECKER
    if pipeline.safety_checker is not None:
        safety_checker = pipeline.safety_checker
        clip_num_channels = safety_checker.config.vision_config.num_channels
        clip_image_size = safety_checker.config.vision_config.image_size
        safety_checker.forward = safety_checker.forward_onnx
        onnx_export(
            pipeline.safety_checker,
            model_args=(
                torch.randn(
                    1,
                    clip_num_channels,
                    clip_image_size,
                    clip_image_size,
                ).to(device=ctx.training_device, dtype=dtype),
                torch.randn(1, vae_sample_size, vae_sample_size, vae_out_channels).to(
                    device=ctx.training_device, dtype=dtype
                ),
            ),
            output_path=output_path / "safety_checker" / "model.onnx",
            ordered_input_names=["clip_input", "images"],
            output_names=["out_images", "has_nsfw_concepts"],
            dynamic_axes={
                "clip_input": {0: "batch", 1: "channels", 2: "height", 3: "width"},
                "images": {0: "batch", 1: "height", 2: "width", 3: "channels"},
            },
            opset=ctx.opset,
        )
        del pipeline.safety_checker
        safety_checker = OnnxRuntimeModel.from_pretrained(
            output_path / "safety_checker"
        )
        feature_extractor = pipeline.feature_extractor
    else:
        safety_checker = None
        feature_extractor = None

    if single_vae:
        onnx_pipeline = OnnxStableDiffusionUpscalePipeline(
            vae=OnnxRuntimeModel.from_pretrained(output_path / "vae"),
            text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
            tokenizer=pipeline.tokenizer,
            low_res_scheduler=pipeline.scheduler,
            unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
            scheduler=pipeline.scheduler,
        )
    else:
        onnx_pipeline = OnnxStableDiffusionPipeline(
            vae_encoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_encoder"),
            vae_decoder=OnnxRuntimeModel.from_pretrained(output_path / "vae_decoder"),
            text_encoder=OnnxRuntimeModel.from_pretrained(output_path / "text_encoder"),
            tokenizer=pipeline.tokenizer,
            unet=OnnxRuntimeModel.from_pretrained(output_path / "unet"),
            scheduler=pipeline.scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            requires_safety_checker=safety_checker is not None,
        )

    logger.info("exporting ONNX model")

    onnx_pipeline.save_pretrained(output_path)
    logger.info("ONNX pipeline saved to %s", output_path)

    del pipeline
    del onnx_pipeline

    if single_vae:
        _ = OnnxStableDiffusionUpscalePipeline.from_pretrained(
            output_path, provider="CPUExecutionProvider"
        )
    else:
        _ = OnnxStableDiffusionPipeline.from_pretrained(
            output_path, provider="CPUExecutionProvider"
        )

    logger.info("ONNX pipeline is loadable")
