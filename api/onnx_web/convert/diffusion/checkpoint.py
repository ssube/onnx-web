###
# Parts of this file are copied or derived from:
#   https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/dreambooth/diff_to_sd.py
#   https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py
#
# Originally by https://github.com/d8ahazard and https://github.com/huggingface
# Those portions *are not* covered by the MIT licensed used for the rest of the onnx-web project.
# In particular, you cannot use this converter for commercial purposes without permission.
#
# d8ahazard code used under No-Commercial License with Limited Commercial Use
#   https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/license.md
# HuggingFace code used under the Apache License, Version 2.0
#   https://github.com/huggingface/diffusers/blob/main/LICENSE
###

import json
import os
import re
import shutil
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PaintByExamplePipeline,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
    LDMBertConfig,
    LDMBertModel,
)
from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils.tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    BertTokenizerFast,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionConfig,
)

from ...utils import load_config, sanitize_name
from ..utils import ConversionContext, load_tensor

logger = getLogger(__name__)


class Config(object):
    """
    Shim for pydantic-style config.
    """

    def __init__(self, kwargs):
        self.__dict__.update(kwargs)
        for k, v in self.__dict__.items():
            Config.config_from_key(self, k, v)

    def __iter__(self):
        for k in self.__dict__.keys():
            yield k

    @classmethod
    def config_from_key(cls, target, k, v):
        if isinstance(v, dict):
            tmp = Config(v)
            setattr(target, k, tmp)
        else:
            setattr(target, k, v)


class TrainingConfig:
    """
    From https://github.com/d8ahazard/sd_dreambooth_extension/blob/main/dreambooth/db_config.py
    """

    adamw_weight_decay: float = 0.01
    attention: str = "default"
    cache_latents: bool = True
    center_crop: bool = True
    freeze_clip_normalization: bool = False
    clip_skip: int = 1
    concepts_list: List[Dict] = []
    concepts_path: str = ""
    custom_model_name: str = ""
    epoch: int = 0
    epoch_pause_frequency: int = 0
    epoch_pause_time: int = 0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_set_to_none: bool = True
    graph_smoothing: int = 50
    half_model: bool = False
    train_unfrozen: bool = False
    has_ema: bool = False
    hflip: bool = False
    initial_revision: int = 0
    learning_rate: float = 5e-6
    learning_rate_min: float = 1e-6
    lifetime_revision: int = 0
    lora_learning_rate: float = 1e-4
    lora_model_name: str = ""
    lora_rank: int = 4
    lora_txt_learning_rate: float = 5e-5
    lora_txt_weight: float = 1.0
    lora_weight: float = 1.0
    lr_cycles: int = 1
    lr_factor: float = 0.5
    lr_power: float = 1.0
    lr_scale_pos: float = 0.5
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_token_length: int = 75
    mixed_precision: str = "fp16"
    model_name: str = ""
    model_dir: str = ""
    model_path: str = ""
    num_train_epochs: int = 100
    pad_tokens: bool = True
    pretrained_model_name_or_path: str = ""
    pretrained_vae_name_or_path: str = ""
    prior_loss_scale: bool = False
    prior_loss_target: int = 100
    prior_loss_weight: float = 1.0
    prior_loss_weight_min: float = 0.1
    resolution: int = 512
    revision: int = 0
    sample_batch_size: int = 1
    sanity_prompt: str = ""
    sanity_seed: int = 420420
    save_ckpt_after: bool = True
    save_ckpt_cancel: bool = False
    save_ckpt_during: bool = True
    save_embedding_every: int = 25
    save_lora_after: bool = True
    save_lora_cancel: bool = False
    save_lora_during: bool = True
    save_preview_every: int = 5
    save_safetensors: bool = False
    save_state_after: bool = False
    save_state_cancel: bool = False
    save_state_during: bool = False
    scheduler: str = "ddim"
    shuffle_tags: bool = False
    snapshot: str = ""
    src: str = ""
    stop_text_encoder: float = 1.0
    train_batch_size: int = 1
    train_imagic: bool = False
    train_unet: bool = True
    use_8bit_adam: bool = True
    use_concepts: bool = False
    use_ema: bool = True
    use_lora: bool = False
    use_subdir: bool = False
    v2: bool = False

    def __init__(
        self,
        conversion: ConversionContext,
        model_name: str = "",
        scheduler: str = "ddim",
        v2: bool = False,
        src: str = "",
        resolution: int = 512,
        **kwargs,
    ):
        model_name = sanitize_name(model_name)
        model_dir = os.path.join(conversion.cache_path, model_name)
        working_dir = os.path.join(model_dir, "working")

        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        self.model_name = model_name
        self.model_dir = model_dir
        self.pretrained_model_name_or_path = working_dir
        self.resolution = resolution
        self.src = src
        self.scheduler = scheduler
        self.v2 = v2

        # avoid pydantic dep for this one fn
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, backup=False):
        """
        Save the config file
        """
        models_path = self.model_dir
        config_file = os.path.join(models_path, "db_config.json")
        if backup:
            backup_dir = os.path.join(models_path, "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            config_file = os.path.join(
                models_path, "backups", f"db_config_{self.revision}.json"
            )
        with open(config_file, "w") as outfile:
            json.dump(self.__dict__, outfile, indent=4)

    def load_params(self, params_dict):
        for key, value in params_dict.items():
            if "db_" in key:
                key = key.replace("db_", "")
            if hasattr(self, key):
                setattr(self, key, value)


# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conversion script for the LDM checkpoints. """


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        new_item = new_item.replace("nin_shortcut", "conv_shortcut")
        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item
        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_vae_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("q.weight", "query.weight")
        new_item = new_item.replace("q.bias", "query.bias")

        new_item = new_item.replace("k.weight", "key.weight")
        new_item = new_item.replace("k.bias", "key.bias")

        new_item = new_item.replace("v.weight", "value.weight")
        new_item = new_item.replace("v.bias", "value.bias")

        new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
        new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

        new_item = shave_segments(
            new_item, n_shave_prefix_segments=n_shave_prefix_segments
        )

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_to_checkpoint(
    paths,
    checkpoint,
    old_checkpoint,
    attention_paths_to_split=None,
    additional_replacements=None,
    config=None,
):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming
    to them. It splits attention layers, and takes into account additional replacements
    that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(
        paths, list
    ), "Paths should be a list of dicts containing 'old' and 'new' keys."

    # Splits the attention layers into three variables.
    if attention_paths_to_split is not None:
        for path, path_map in attention_paths_to_split.items():
            old_tensor = old_checkpoint[path]
            channels = old_tensor.shape[0] // 3

            target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

            num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

            old_tensor = old_tensor.reshape(
                (num_heads, 3 * channels // num_heads) + old_tensor.shape[1:]
            )
            query, key, value = old_tensor.split(channels // num_heads, dim=1)

            checkpoint[path_map["query"]] = query.reshape(target_shape)
            checkpoint[path_map["key"]] = key.reshape(target_shape)
            checkpoint[path_map["value"]] = value.reshape(target_shape)

    for path in paths:
        new_path = path["new"]

        # These have already been assigned
        if (
            attention_paths_to_split is not None
            and new_path in attention_paths_to_split
        ):
            continue

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


def conv_attn_to_linear(checkpoint):
    keys = list(checkpoint.keys())
    attn_keys = ["query.weight", "key.weight", "value.weight"]
    for key in keys:
        if ".".join(key.split(".")[-2:]) in attn_keys:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0, 0]
        elif "proj_attn.weight" in key:
            if checkpoint[key].ndim > 2:
                checkpoint[key] = checkpoint[key][:, :, 0]


def create_unet_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    unet_params = original_config.model.params.unet_config.params
    vae_params = original_config.model.params.first_stage_config.params.ddconfig

    block_out_channels = [
        unet_params.model_channels * mult for mult in unet_params.channel_mult
    ]

    down_block_types = []
    resolution = 1
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnDownBlock2D"
            if resolution in unet_params.attention_resolutions
            else "DownBlock2D"
        )
        down_block_types.append(block_type)
        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        block_type = (
            "CrossAttnUpBlock2D"
            if resolution in unet_params.attention_resolutions
            else "UpBlock2D"
        )
        up_block_types.append(block_type)
        resolution //= 2

    vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

    head_dim = unet_params.num_heads if "num_heads" in unet_params else None
    use_linear_projection = (
        unet_params.use_linear_in_transformer
        if "use_linear_in_transformer" in unet_params
        else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim = [5, 10, 20, 20]

    config = dict(
        sample_size=image_size // vae_scale_factor,
        in_channels=unet_params.in_channels,
        out_channels=unet_params.out_channels,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        layers_per_block=unet_params.num_res_blocks,
        cross_attention_dim=unet_params.context_dim,
        attention_head_dim=head_dim,
        use_linear_projection=use_linear_projection,
    )

    return config


def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    vae_params = original_config.model.params.first_stage_config.params.ddconfig
    _ = original_config.model.params.first_stage_config.params.embed_dim

    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = dict(
        sample_size=image_size,
        in_channels=vae_params.in_channels,
        out_channels=vae_params.out_ch,
        down_block_types=tuple(down_block_types),
        up_block_types=tuple(up_block_types),
        block_out_channels=tuple(block_out_channels),
        latent_channels=vae_params.z_channels,
        layers_per_block=vae_params.num_res_blocks,
    )
    return config


def create_diffusers_schedular(original_config):
    schedular = DDIMScheduler(
        num_train_timesteps=original_config.model.params.timesteps,
        beta_start=original_config.model.params.linear_start,
        beta_end=original_config.model.params.linear_end,
        beta_schedule="scaled_linear",
    )
    return schedular


def create_ldm_bert_config(original_config):
    bert_params = original_config.model.parms.cond_stage_config.params
    config = LDMBertConfig(
        d_model=bert_params.n_embed,
        encoder_layers=bert_params.n_layer,
        encoder_ffn_dim=bert_params.n_embed * 4,
    )
    return config


def convert_ldm_unet_checkpoint(checkpoint, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """

    # extract state_dict for UNet
    unet_state_dict = {}
    keys = list(checkpoint.keys())
    has_ema = False
    unet_key = "model.diffusion_model."
    # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
    if sum(k.startswith("model_ema") for k in keys) > 100:
        print(f"Checkpoint {path} has both EMA and non-EMA weights.")
        if extract_ema:
            has_ema = True
            print(
                "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
            )
            for key in keys:
                if key.startswith("model.diffusion_model"):
                    flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                    unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(
                        flat_ema_key
                    )
        else:
            print(
                "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
            )

    for key in keys:
        if key.startswith(unet_key):
            unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

    new_checkpoint = {
        "time_embedding.linear_1.weight": unet_state_dict["time_embed.0.weight"],
        "time_embedding.linear_1.bias": unet_state_dict["time_embed.0.bias"],
        "time_embedding.linear_2.weight": unet_state_dict["time_embed.2.weight"],
        "time_embedding.linear_2.bias": unet_state_dict["time_embed.2.bias"],
        "conv_in.weight": unet_state_dict["input_blocks.0.0.weight"],
        "conv_in.bias": unet_state_dict["input_blocks.0.0.bias"],
        "conv_norm_out.weight": unet_state_dict["out.0.weight"],
        "conv_norm_out.bias": unet_state_dict["out.0.bias"],
        "conv_out.weight": unet_state_dict["out.2.weight"],
        "conv_out.bias": unet_state_dict["out.2.bias"],
    }

    # Retrieves the keys for the input blocks only
    num_input_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "input_blocks" in layer
        }
    )
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "middle_block" in layer
        }
    )
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len(
        {
            ".".join(layer.split(".")[:2])
            for layer in unet_state_dict
            if "output_blocks" in layer
        }
    )
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key
            for key in input_blocks[i]
            if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.weight"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.weight")
            new_checkpoint[
                f"down_blocks.{block_id}.downsamplers.0.conv.bias"
            ] = unet_state_dict.pop(f"input_blocks.{i}.0.op.bias")

        paths = renew_resnet_paths(resnets)
        meta_path = {
            "old": f"input_blocks.{i}.0",
            "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}",
        }
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            unet_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

        if len(attentions):
            paths = renew_attention_paths(attentions)
            meta_path = {
                "old": f"input_blocks.{i}.1",
                "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths,
        new_checkpoint,
        unet_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [
                key for key in output_blocks[i] if f"output_blocks.{i}.1" in key
            ]

            resnet_0_paths = renew_resnet_paths(resnets)
            paths = renew_resnet_paths(resnets)

            meta_path = {
                "old": f"output_blocks.{i}.0",
                "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}",
            }
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(
                    ["conv.bias", "conv.weight"]
                )
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.weight"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.weight"]
                new_checkpoint[
                    f"up_blocks.{block_id}.upsamplers.0.conv.bias"
                ] = unet_state_dict[f"output_blocks.{i}.{index}.conv.bias"]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": f"output_blocks.{i}.1",
                    "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                }
                assign_to_checkpoint(
                    paths,
                    new_checkpoint,
                    unet_state_dict,
                    additional_replacements=[meta_path],
                    config=config,
                )
        else:
            resnet_0_paths = renew_resnet_paths(
                output_block_layers, n_shave_prefix_segments=1
            )
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(
                    [
                        "up_blocks",
                        str(block_id),
                        "resnets",
                        str(layer_in_block_id),
                        path["new"],
                    ]
                )

                new_checkpoint[new_path] = unet_state_dict[old_path]

    # From Bmalthais
    # if v2:
    # linear_transformer_to_conv(new_checkpoint)
    return new_checkpoint, has_ema


def convert_ldm_vae_checkpoint(checkpoint, config, first_stage=True):
    # extract state dict for VAE
    vae_state_dict = {}
    vae_key = "first_stage_model."
    keys = list(checkpoint.keys())
    for key in keys:
        if first_stage:
            if key.startswith(vae_key):
                vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)
        else:
            vae_state_dict[key] = checkpoint.get(key)

    new_checkpoint = {
        "encoder.conv_in.weight": vae_state_dict["encoder.conv_in.weight"],
        "encoder.conv_in.bias": vae_state_dict["encoder.conv_in.bias"],
        "encoder.conv_out.weight": vae_state_dict["encoder.conv_out.weight"],
        "encoder.conv_out.bias": vae_state_dict["encoder.conv_out.bias"],
        "encoder.conv_norm_out.weight": vae_state_dict["encoder.norm_out.weight"],
        "encoder.conv_norm_out.bias": vae_state_dict["encoder.norm_out.bias"],
        "decoder.conv_in.weight": vae_state_dict["decoder.conv_in.weight"],
        "decoder.conv_in.bias": vae_state_dict["decoder.conv_in.bias"],
        "decoder.conv_out.weight": vae_state_dict["decoder.conv_out.weight"],
        "decoder.conv_out.bias": vae_state_dict["decoder.conv_out.bias"],
        "decoder.conv_norm_out.weight": vae_state_dict["decoder.norm_out.weight"],
        "decoder.conv_norm_out.bias": vae_state_dict["decoder.norm_out.bias"],
        "quant_conv.weight": vae_state_dict["quant_conv.weight"],
        "quant_conv.bias": vae_state_dict["quant_conv.bias"],
        "post_quant_conv.weight": vae_state_dict["post_quant_conv.weight"],
        "post_quant_conv.bias": vae_state_dict["post_quant_conv.bias"],
    }

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "encoder.down" in layer
        }
    )
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key]
        for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len(
        {
            ".".join(layer.split(".")[:3])
            for layer in vae_state_dict
            if "decoder.up" in layer
        }
    )
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key]
        for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [
            key
            for key in down_blocks[i]
            if f"down.{i}" in key and f"down.{i}.downsample" not in key
        ]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.weight")
            new_checkpoint[
                f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"
            ] = vae_state_dict.pop(f"encoder.down.{i}.downsample.conv.bias")

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key
            for key in up_blocks[block_id]
            if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.weight"]
            new_checkpoint[
                f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"
            ] = vae_state_dict[f"decoder.up.{block_id}.upsample.conv.bias"]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(
            paths,
            new_checkpoint,
            vae_state_dict,
            additional_replacements=[meta_path],
            config=config,
        )

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        paths,
        new_checkpoint,
        vae_state_dict,
        additional_replacements=[meta_path],
        config=config,
    )
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint


def convert_ldm_bert_checkpoint(checkpoint, config):
    def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
        hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
        hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
        hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight

        hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
        hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

    def _copy_linear(hf_linear, pt_linear):
        hf_linear.weight = pt_linear.weight
        hf_linear.bias = pt_linear.bias

    def _copy_layer(hf_layer, pt_layer):
        # copy layer norms
        _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
        _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])

        # copy attn
        _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])

        # copy MLP
        pt_mlp = pt_layer[1][1]
        _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
        _copy_linear(hf_layer.fc2, pt_mlp.net[2])

    def _copy_layers(hf_layers, pt_layers):
        for i, hf_layer in enumerate(hf_layers):
            if i != 0:
                i += i
            pt_layer = pt_layers[i : i + 2]
            _copy_layer(hf_layer, pt_layer)

    hf_model = LDMBertModel(config).eval()

    # copy  embeds
    hf_model.model.embed_tokens.weight = checkpoint.transformer.token_emb.weight
    hf_model.model.embed_positions.weight.data = (
        checkpoint.transformer.pos_emb.emb.weight
    )

    # copy layer norm
    _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)

    # copy hidden layers
    _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.layers)

    _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)

    return hf_model


def convert_ldm_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    keys = list(checkpoint.keys())

    text_model_dict = {}

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            if key.find("text_model") == -1:
                text_model_dict[
                    "text_model." + key[len("cond_stage_model.transformer.") :]
                ] = checkpoint[key]
            else:
                text_model_dict[
                    key[len("cond_stage_model.transformer.") :]
                ] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model


textenc_conversion_lst = [
    (
        "cond_stage_model.model.positional_embedding",
        "text_model.embeddings.position_embedding.weight",
    ),
    (
        "cond_stage_model.model.token_embedding.weight",
        "text_model.embeddings.token_embedding.weight",
    ),
    ("cond_stage_model.model.ln_final.weight", "text_model.final_layer_norm.weight"),
    ("cond_stage_model.model.ln_final.bias", "text_model.final_layer_norm.bias"),
]
textenc_conversion_map = {x[0]: x[1] for x in textenc_conversion_lst}

textenc_transformer_conversion_lst = [
    # (stable-diffusion, HF Diffusers)
    ("resblocks.", "text_model.encoder.layers."),
    ("ln_1", "layer_norm1"),
    ("ln_2", "layer_norm2"),
    (".c_fc.", ".fc1."),
    (".c_proj.", ".fc2."),
    (".attn", ".self_attn"),
    ("ln_final.", "transformer.text_model.final_layer_norm."),
    (
        "token_embedding.weight",
        "transformer.text_model.embeddings.token_embedding.weight",
    ),
    (
        "positional_embedding",
        "transformer.text_model.embeddings.position_embedding.weight",
    ),
]
protected = {re.escape(x[0]): x[1] for x in textenc_transformer_conversion_lst}
textenc_pattern = re.compile("|".join(protected.keys()))


def convert_paint_by_example_checkpoint(checkpoint):
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
    model = PaintByExampleImageEncoder(config)

    keys = list(checkpoint.keys())

    text_model_dict = {}

    for key in keys:
        if key.startswith("cond_stage_model.transformer"):
            text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[
                key
            ]

    # load clip vision
    model.model.load_state_dict(text_model_dict)

    # load mapper
    keys_mapper = {
        k[len("cond_stage_model.mapper.res") :]: v
        for k, v in checkpoint.items()
        if k.startswith("cond_stage_model.mapper")
    }

    MAPPING = {
        "attn.c_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
        "attn.c_proj": ["attn1.to_out.0"],
        "ln_1": ["norm1"],
        "ln_2": ["norm3"],
        "mlp.c_fc": ["ff.net.0.proj"],
        "mlp.c_proj": ["ff.net.2"],
    }

    mapped_weights = {}
    for key, value in keys_mapper.items():
        prefix = key[: len("blocks.i")]
        suffix = key.split(prefix)[-1].split(".")[-1]
        name = key.split(prefix)[-1].split(suffix)[0][1:-1]
        mapped_names = MAPPING[name]

        num_splits = len(mapped_names)
        for i, mapped_name in enumerate(mapped_names):
            new_name = ".".join([prefix, mapped_name, suffix])
            shape = value.shape[0] // num_splits
            mapped_weights[new_name] = value[i * shape : (i + 1) * shape]

    model.mapper.load_state_dict(mapped_weights)

    # load final layer norm
    model.final_layer_norm.load_state_dict(
        {
            "bias": checkpoint["cond_stage_model.final_ln.bias"],
            "weight": checkpoint["cond_stage_model.final_ln.weight"],
        }
    )

    # load final proj
    model.proj_out.load_state_dict(
        {
            "bias": checkpoint["proj_out.bias"],
            "weight": checkpoint["proj_out.weight"],
        }
    )

    # load uncond vector
    model.uncond_vector.data = torch.nn.Parameter(checkpoint["learnable_vector"])
    return model


def convert_open_clip_checkpoint(checkpoint):
    text_model = CLIPTextModel.from_pretrained(
        "stabilityai/stable-diffusion-2", subfolder="text_encoder"
    )

    keys = list(checkpoint.keys())
    text_model_dict = {}
    if "cond_stage_model.model.text_projection" in checkpoint:
        d_model = int(checkpoint["cond_stage_model.model.text_projection"].shape[0])
    else:
        logger.debug("no projection shape found, setting to 1024")
        d_model = 1024
    text_model_dict[
        "text_model.embeddings.position_ids"
    ] = text_model.text_model.embeddings.get_buffer("position_ids")

    for key in keys:
        if (
            "resblocks.23" in key
        ):  # Diffusers drops the final layer and only uses the penultimate layer
            continue
        if key in textenc_conversion_map:
            text_model_dict[textenc_conversion_map[key]] = checkpoint[key]
        if key.startswith("cond_stage_model.model.transformer."):
            new_key = key[len("cond_stage_model.model.transformer.") :]
            if new_key.endswith(".in_proj_weight"):
                new_key = new_key[: -len(".in_proj_weight")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.weight"] = checkpoint[key][
                    :d_model, :
                ]
                text_model_dict[new_key + ".k_proj.weight"] = checkpoint[key][
                    d_model : d_model * 2, :
                ]
                text_model_dict[new_key + ".v_proj.weight"] = checkpoint[key][
                    d_model * 2 :, :
                ]
            elif new_key.endswith(".in_proj_bias"):
                new_key = new_key[: -len(".in_proj_bias")]
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )
                text_model_dict[new_key + ".q_proj.bias"] = checkpoint[key][:d_model]
                text_model_dict[new_key + ".k_proj.bias"] = checkpoint[key][
                    d_model : d_model * 2
                ]
                text_model_dict[new_key + ".v_proj.bias"] = checkpoint[key][
                    d_model * 2 :
                ]
            else:
                new_key = textenc_pattern.sub(
                    lambda m: protected[re.escape(m.group(0))], new_key
                )

                text_model_dict[new_key] = checkpoint[key]

    text_model.load_state_dict(text_model_dict)

    return text_model


def replace_symlinks(path, base):
    if os.path.islink(path):
        # Get the target of the symlink
        src = os.readlink(path)
        blob = os.path.basename(src)
        path_parts = path.split("/") if "/" in path else path.split("\\")
        model_name = None
        dir_name = None
        save_next = False
        for part in path_parts:
            if save_next:
                model_name = part
                break
            if part == "src" or part == "working":
                dir_name = part
                save_next = True
        if model_name is not None and dir_name is not None:
            blob_path = os.path.join(base, dir_name, model_name, "blobs", blob)
        else:
            blob_path = None

        if blob_path is None:
            logger.debug("no blob")
            return
        os.replace(blob_path, path)
    elif os.path.isdir(path):
        # Recursively replace symlinks in the directory
        for subpath in os.listdir(path):
            replace_symlinks(os.path.join(path, subpath), base)


def download_model(db_config: TrainingConfig, token):
    hub_url = db_config.src
    if "http" in hub_url or "huggingface.co" in hub_url:
        hub_url = "/".join(hub_url.split("/")[-2:])

    api = HfApi()
    repo_info = api.repo_info(
        repo_id=hub_url,
        repo_type="model",
        revision="main",
        token=token,
    )

    if repo_info.sha is None:
        logger.warning("unable to fetch repo info: %s", hub_url)
        return None, None

    siblings = repo_info.siblings

    diffusion_dirs = [
        "text_encoder",
        "unet",
        "vae",
        "tokenizer",
        "scheduler",
        "feature_extractor",
        "safety_checker",
    ]
    config_file = None
    model_index = None
    model_files = []
    diffusion_files = []

    for sibling in siblings:
        name = sibling.rfilename
        if "inference.yaml" in name:
            config_file = name
            continue
        if "model_index.json" in name:
            model_index = name
            continue
        if (".ckpt" in name or ".safetensors" in name) and "/" not in name:
            model_files.append(name)
            continue
        for diffusion_dir in diffusion_dirs:
            if f"{diffusion_dir}/" in name:
                diffusion_files.append(name)

    for diffusion_dir in diffusion_dirs:
        safe_model = None
        bin_model = None
        for diffusion_file in diffusion_files:
            if diffusion_dir in diffusion_file:
                if ".safetensors" in diffusion_file:
                    safe_model = diffusion_file
                if ".bin" in diffusion_file:
                    bin_model = diffusion_file
        if safe_model and bin_model:
            diffusion_files.remove(bin_model)

    model_file = next(
        (x for x in model_files if ".safetensors" in x and "nonema" in x),
        next(
            (x for x in model_files if "nonema" in x),
            next(
                (x for x in model_files if ".safetensors" in x),
                model_files[0] if model_files else None,
            ),
        ),
    )

    files_to_fetch = None

    if model_file is not None:
        files_to_fetch = [model_file]
    elif len(diffusion_files):
        files_to_fetch = diffusion_files
        if model_index is not None:
            files_to_fetch.append(model_index)

    if files_to_fetch and config_file:
        files_to_fetch.append(config_file)

    logger.info(f"Fetching files: {files_to_fetch}")

    if not len(files_to_fetch):
        logger.debug("nothing to fetch")
        return None, None

    out_model = None
    for repo_file in tqdm(files_to_fetch, desc=f"Fetching {len(files_to_fetch)} files"):
        out = hf_hub_download(
            hub_url,
            filename=repo_file,
            repo_type="model",
            revision=repo_info.sha,
            token=token,
        )
        replace_symlinks(out, db_config.model_dir)
        dest = None
        file_name = os.path.basename(out)
        if "yaml" in repo_file:
            dest = os.path.join(db_config.model_dir)
        if "model_index" in repo_file:
            dest = db_config.pretrained_model_name_or_path
        if not dest:
            for diffusion_dir in diffusion_dirs:
                if diffusion_dir in out:
                    out_model = db_config.pretrained_model_name_or_path
                    dest = os.path.join(
                        db_config.pretrained_model_name_or_path, diffusion_dir
                    )
        if not dest:
            if ".ckpt" in out or ".safetensors" in out:
                dest = os.path.join(db_config.model_dir, "src")
                out_model = dest

        if dest is not None:
            if not os.path.exists(dest):
                os.makedirs(dest)
            dest_file = os.path.join(dest, file_name)
            if os.path.exists(dest_file):
                os.remove(dest_file)
            shutil.copyfile(out, dest_file)

    return out_model, config_file


def get_config_path(
    conversion: ConversionContext,
    model_version: str = "v1",
    train_type: str = "default",
    config_base_name: str = "training",
    prediction_type: str = "epsilon",
):
    train_type = (
        f"{train_type}" if not prediction_type == "v_prediction" else f"{train_type}-v"
    )

    parts = os.path.join(
        conversion.model_path,
        "configs",
        f"{model_version}-{config_base_name}-{train_type}.yaml",
    )
    return os.path.abspath(parts)


def get_config_file(
    conversion: ConversionContext,
    train_unfrozen=False,
    v2=False,
    prediction_type="epsilon",
    config_file=None,
):
    if config_file is not None:
        return config_file

    config_base_name = "training"

    model_versions = {"v1": "v1", "v2": "v2"}
    train_types = {
        "default": "default",
        "unfrozen": "unfrozen",
    }

    model_train_type = train_types["default"]
    model_version_name = f"{model_versions['v1'] if not v2 else model_versions['v2']}"

    if train_unfrozen:
        model_train_type = train_types["unfrozen"]
    else:
        model_train_type = train_types["default"]

    return get_config_path(
        conversion,
        model_version_name,
        model_train_type,
        config_base_name,
        prediction_type,
    )


def extract_checkpoint(
    conversion: ConversionContext,
    new_model_name: str,
    checkpoint_file: str,
    is_inpainting: Optional[bool] = False,
    scheduler_type="ddim",
    extract_ema=False,
    train_unfrozen=False,
    is_512=True,
    config_file: str = None,
    vae_file: str = None,
):
    """

    @param new_model_name: The name of the new model
    @param checkpoint_file: The source checkpoint to use, if not from hub. Needs full path
    @param scheduler_type: The target scheduler type
    @param from_hub: Are we making this model from the hub?
    @param new_model_url: The URL to pull. Should be formatted like compviz/stable-diffusion-2, not a full URL.
    @param new_model_token: Your huggingface.co token.
    @param extract_ema: Whether to extract EMA weights if present.
    @param is_512: Is it a 512 model?
    @return:
        db_new_model_name: Gr.dropdown populated with our model name, if applicable.
        db_config.model_dir: The directory where our model was created.
        db_config.revision: Model revision
        db_config.epoch: Model epoch
        db_config.scheduler: The scheduler being used
        db_config.src: The source checkpoint, if not from hub.
        db_has_ema: Whether the model had EMA weights and they were extracted. If weights were not present or
        you did not extract them and they were, this will be false.
        db_resolution: The resolution the model trains at.
        db_v2: Is this a V2 Model?

        status
    """
    has_ema = False
    v2 = False
    revision = 0
    epoch = 0
    image_size = 512 if is_512 else 768
    # Needed for V2 models so we can create the right text encoder.
    upcast_attention = False

    # Create empty config
    db_config = TrainingConfig(
        conversion,
        model_name=new_model_name,
        scheduler=scheduler_type,
        src=checkpoint_file,
    )

    original_config_file = None

    try:
        map_location = torch.device("cpu")

        # Try to determine if v1 or v2 model if we have a ckpt
        logger.info("loading model from checkpoint")
        checkpoint = load_tensor(checkpoint_file, map_location=map_location)
        if checkpoint is None:
            logger.warning("unable to load tensor")
            return False

        rev_keys = ["db_global_step", "global_step"]
        epoch_keys = ["db_epoch", "epoch"]
        for key in rev_keys:
            if key in checkpoint:
                revision = checkpoint[key]
                break

        for key in epoch_keys:
            if key in checkpoint:
                epoch = checkpoint[key]
                break

        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
            if not is_512:
                # v2.1 needs to upcast attention
                logger.debug("setting upcast_attention")
                upcast_attention = True
            v2 = True
        else:
            v2 = False

        if v2 and not is_512:
            prediction_type = "v_prediction"
        else:
            prediction_type = "epsilon"

        original_config_file = get_config_file(
            conversion, train_unfrozen, v2, prediction_type, config_file=config_file
        )

        logger.info(
            f"Pred and size are {prediction_type} and {image_size}, using config: {original_config_file}"
        )
        db_config.resolution = image_size
        db_config.lifetime_revision = revision
        db_config.epoch = epoch
        db_config.v2 = v2

        logger.info(f"{'v2' if v2 else 'v1'} model loaded.")

        # Use existing YAML if present
        if checkpoint_file is not None:
            config_check = (
                checkpoint_file.replace(".ckpt", ".yaml")
                if ".ckpt" in checkpoint_file
                else checkpoint_file.replace(".safetensors", ".yaml")
            )
            if os.path.exists(config_check):
                original_config_file = config_check

        if original_config_file is None or not os.path.exists(original_config_file):
            logger.warning(
                "unable to select a config file: %s" % (original_config_file)
            )
            return False

        logger.debug("trying to load: %s", original_config_file)
        original_config = Config(load_config(original_config_file))

        num_train_timesteps = original_config.model.params.timesteps
        beta_start = original_config.model.params.linear_start
        beta_end = original_config.model.params.linear_end

        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
        # make sure scheduler works correctly with DDIM
        scheduler.register_to_config(clip_sample=False)
        if scheduler_type == "pndm":
            config = dict(scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(config)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        elif scheduler_type == "ddim":
            pass
        else:
            raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

        # Convert the UNet2DConditionModel model.
        logger.info("converting UNet")
        unet_config = create_unet_diffusers_config(
            original_config, image_size=image_size
        )
        unet_config["upcast_attention"] = upcast_attention
        if is_inpainting:
            unet_config["in_channels"] = 9

        unet = UNet2DConditionModel(**unet_config)

        converted_unet_checkpoint, has_ema = convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=checkpoint_file, extract_ema=extract_ema
        )
        db_config.has_ema = has_ema
        db_config.save()
        unet.load_state_dict(converted_unet_checkpoint)

        # Convert the VAE model.
        logger.info("converting VAE")
        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)

        if vae_file is None:
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                checkpoint, vae_config
            )
        else:
            vae_file = os.path.join(conversion.model_path, vae_file)
            logger.debug("loading custom VAE: %s", vae_file)
            vae_checkpoint = load_tensor(vae_file, map_location=map_location)
            converted_vae_checkpoint = convert_ldm_vae_checkpoint(
                vae_checkpoint, vae_config, first_stage=False
            )

        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)

        # Convert the text model.
        logger.info("converting text encoder")
        text_model_type = original_config.model.params.cond_stage_config.target.split(
            "."
        )[-1]
        if text_model_type == "FrozenOpenCLIPEmbedder":
            text_model = convert_open_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained(
                "stabilityai/stable-diffusion-2", subfolder="tokenizer"
            )
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        elif text_model_type == "PaintByExample":
            vision_model = convert_paint_by_example_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            )
            pipe = PaintByExamplePipeline(
                vae=vae,
                image_encoder=vision_model,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
            )
        elif text_model_type == "FrozenCLIPEmbedder":
            text_model = convert_ldm_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            safety_checker = StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            )
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            )
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
        else:
            text_config = create_ldm_bert_config(original_config)
            text_model = convert_ldm_bert_checkpoint(checkpoint, text_config)
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            pipe = LDMTextToImagePipeline(
                vqvae=vae,
                bert=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
            )
    except Exception:
        logger.exception(
            "error setting up output",
        )
        pipe = None

    if pipe is None or db_config is None:
        logger.error("pipeline or config is not set, unable to continue")
        return False
    else:
        logger.info("saving diffusion model")
        pipe.save_pretrained(db_config.pretrained_model_name_or_path)
        result_status = f"Checkpoint successfully extracted to {db_config.pretrained_model_name_or_path}"
        revision = db_config.revision
        scheduler = db_config.scheduler
        required_dirs = ["unet", "vae", "text_encoder", "scheduler", "tokenizer"]
        if original_config_file is not None and os.path.exists(original_config_file):
            logger.debug(
                "copying original config: %s -> %s",
                original_config_file,
                db_config.model_dir,
            )
            shutil.copy(original_config_file, db_config.model_dir)
            basename = os.path.basename(original_config_file)
            new_ex_path = os.path.join(db_config.model_dir, basename)
            new_name = os.path.join(db_config.model_dir, f"{db_config.model_name}.yaml")
            logger.debug(
                "copying model config to new name: %s -> %s", new_ex_path, new_name
            )
            if os.path.exists(new_name):
                os.remove(new_name)
            os.rename(new_ex_path, new_name)

        for req_dir in required_dirs:
            full_path = os.path.join(db_config.pretrained_model_name_or_path, req_dir)
            if not os.path.exists(full_path):
                result_status = f"Missing model directory, removing model: {full_path}"
                shutil.rmtree(db_config.model_dir, ignore_errors=False, onerror=None)
                return False

        remove_dirs = ["logging", "samples"]
        for rd in remove_dirs:
            rem_dir = os.path.join(db_config.model_dir, rd)
            if os.path.exists(rem_dir):
                shutil.rmtree(rem_dir, True)
                if not os.path.exists(rem_dir):
                    os.makedirs(rem_dir)

    logger.info(result_status)
    return True


@torch.no_grad()
def convert_extract_checkpoint(
    conversion: ConversionContext,
    source: str,
    dest: str,
    is_inpainting: Optional[bool] = False,
    config_file: Optional[str] = None,
    vae_file: Optional[str] = None,
) -> Tuple[bool, str]:
    working_name = os.path.join(conversion.cache_path, dest, "working")
    model_index = os.path.join(working_name, "model_index.json")

    if os.path.exists(working_name) and os.path.exists(model_index):
        logger.info("extracted Torch model already exists, reusing: %s", working_name)
    else:
        logger.info(
            "extracting checkpoint to Torch model: %s -> %s",
            source,
            dest,
        )
        if extract_checkpoint(
            conversion,
            dest,
            source,
            config_file=config_file,
            is_inpainting=is_inpainting,
            vae_file=vae_file,
        ):
            logger.info("extracted checkpoint to Torch model: %s", working_name)
        else:
            logger.error("unable to convert checkpoint to Torch model")
            raise ValueError("unable to convert checkpoint to Torch model")

    return working_name
