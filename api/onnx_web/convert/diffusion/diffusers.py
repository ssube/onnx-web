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
from os import mkdir, path
from pathlib import Path
from shutil import rmtree
from typing import Dict, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    OnnxRuntimeModel,
    OnnxStableDiffusionPipeline,
    StableDiffusionPipeline,
)
from onnx import load_model, save_model

from ...constants import ONNX_MODEL, ONNX_WEIGHTS
from ...diffusers.load import optimize_pipeline
from ...diffusers.pipelines.upscale import OnnxStableDiffusionUpscalePipeline
from ...diffusers.version_safe_diffusers import AttnProcessor
from ...models.cnet import UNet2DConditionModel_CNet
from ..utils import ConversionContext, is_torch_2_0, onnx_export

logger = getLogger(__name__)


def convert_diffusion_diffusers_cnet(
    conversion: ConversionContext,
    source: str,
    device: str,
    output_path: Path,
    dtype,
    unet_in_channels,
    unet_sample_size,
    num_tokens,
    text_hidden_size,
):
    # CNet
    pipe_cnet = UNet2DConditionModel_CNet.from_pretrained(source, subfolder="unet").to(
        device=device, dtype=dtype
    )

    if is_torch_2_0:
        pipe_cnet.set_attn_processor(AttnProcessor())

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
        output_path=cnet_path,
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
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            "timestep": {0: "batch"},
            "encoder_hidden_states": {0: "batch", 1: "sequence"},
            "down_block_0": {0: "batch", 2: "height", 3: "width"},
            "down_block_1": {0: "batch", 2: "height", 3: "width"},
            "down_block_2": {0: "batch", 2: "height", 3: "width"},
            "down_block_3": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_4": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_5": {0: "batch", 2: "height2", 3: "width2"},
            "down_block_6": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_7": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_8": {0: "batch", 2: "height4", 3: "width4"},
            "down_block_9": {0: "batch", 2: "height8", 3: "width8"},
            "down_block_10": {0: "batch", 2: "height8", 3: "width8"},
            "down_block_11": {0: "batch", 2: "height8", 3: "width8"},
            "mid_block_additional_residual": {0: "batch", 2: "height8", 3: "width8"},
        },
        opset=conversion.opset,
        half=conversion.half,
        external_data=True,  # UNet is > 2GB, so the weights need to be split
    )
    cnet_model_path = str(cnet_path.absolute().as_posix())
    cnet_dir = path.dirname(cnet_model_path)
    cnet = load_model(cnet_model_path)

    # clean up existing tensor files
    rmtree(cnet_dir)
    mkdir(cnet_dir)

    # collate external tensor files into one
    save_model(
        cnet,
        cnet_model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=ONNX_WEIGHTS,
        convert_attribute=False,
    )
    del pipe_cnet


@torch.no_grad()
def convert_diffusion_diffusers(
    conversion: ConversionContext,
    model: Dict,
    source: str,
) -> Tuple[bool, str]:
    """
    From https://github.com/huggingface/diffusers/blob/main/scripts/convert_stable_diffusion_checkpoint_to_onnx.py
    """
    name = model.get("name")
    source = source or model.get("source")
    single_vae = model.get("single_vae")
    replace_vae = model.get("vae")

    device = conversion.training_device
    dtype = conversion.torch_dtype()
    logger.debug("using Torch dtype %s for pipeline", dtype)

    dest_path = path.join(conversion.model_path, name)
    model_index = path.join(dest_path, "model_index.json")
    model_cnet = path.join(dest_path, "cnet", ONNX_MODEL)

    # diffusers go into a directory rather than .onnx file
    logger.info(
        "converting Stable Diffusion model %s: %s -> %s/", name, source, dest_path
    )

    if single_vae:
        logger.info("converting model with single VAE")

    cnet_only = False
    if path.exists(dest_path) and path.exists(model_index):
        if not path.exists(model_cnet):
            logger.info(
                "ONNX model was converted without a ControlNet UNet, converting one"
            )
            cnet_only = True
        else:
            logger.info("ONNX model already exists, skipping")
            return (False, dest_path)

    pipeline = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=dtype,
        use_auth_token=conversion.token,
    ).to(device)
    output_path = Path(dest_path)

    optimize_pipeline(conversion, pipeline)

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
            output_path=output_path / "text_encoder" / ONNX_MODEL,
            ordered_input_names=["input_ids"],
            output_names=["last_hidden_state", "pooler_output", "hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
            },
            opset=conversion.opset,
            half=conversion.half,
        )

    del pipeline.text_encoder

    # UNET
    logger.debug("UNET config: %s", pipeline.unet.config)
    if single_vae:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "class_labels"]
        unet_scale = torch.tensor(4).to(device=device, dtype=torch.long)
    else:
        unet_inputs = ["sample", "timestep", "encoder_hidden_states", "return_dict"]
        unet_scale = torch.tensor(False).to(device=device, dtype=torch.bool)

    if is_torch_2_0:
        pipeline.unet.set_attn_processor(AttnProcessor())

    unet_in_channels = pipeline.unet.config.in_channels
    unet_sample_size = pipeline.unet.config.sample_size
    unet_path = output_path / "unet" / ONNX_MODEL

    if not cnet_only:
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
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = path.dirname(unet_model_path)
        unet = load_model(unet_model_path)

        # clean up existing tensor files
        rmtree(unet_dir)
        mkdir(unet_dir)

        # collate external tensor files into one
        save_model(
            unet,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=ONNX_WEIGHTS,
            convert_attribute=False,
        )

    del pipeline.unet

    convert_diffusion_diffusers_cnet(
        conversion,
        source,
        device,
        output_path,
        dtype,
        unet_in_channels,
        unet_sample_size,
        num_tokens,
        text_hidden_size,
    )

    if cnet_only:
        logger.info("done converting CNet")
        return (True, dest_path)

    # VAE
    if replace_vae is not None:
        logger.debug("loading custom VAE: %s", replace_vae)
        vae = AutoencoderKL.from_pretrained(replace_vae)
        pipeline.vae = vae

    if single_vae:
        logger.debug("VAE config: %s", pipeline.vae.config)

        # SINGLE VAE
        vae_only = pipeline.vae
        vae_latent_channels = vae_only.config.latent_channels
        # forward only through the decoder part
        vae_only.forward = vae_only.decode
        onnx_export(
            vae_only,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae" / ONNX_MODEL,
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
        onnx_export(
            vae_encoder,
            model_args=(
                torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(
                    device=device, dtype=dtype
                ),
                False,
            ),
            output_path=output_path / "vae_encoder" / ONNX_MODEL,
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
        onnx_export(
            vae_decoder,
            model_args=(
                torch.randn(
                    1, vae_latent_channels, unet_sample_size, unet_sample_size
                ).to(device=device, dtype=dtype),
                False,
            ),
            output_path=output_path / "vae_decoder" / ONNX_MODEL,
            ordered_input_names=["latent_sample", "return_dict"],
            output_names=["sample"],
            dynamic_axes={
                "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
            },
            opset=conversion.opset,
            half=conversion.half,
        )

    del pipeline.vae

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
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
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

    return (True, dest_path)
