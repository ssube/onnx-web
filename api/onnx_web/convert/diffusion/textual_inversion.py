from logging import getLogger
from os import makedirs, path
from typing import Optional

import torch
from huggingface_hub.file_download import hf_hub_download
from torch.onnx import export
from transformers import CLIPTextModel, CLIPTokenizer

from ..utils import ConversionContext

logger = getLogger(__name__)


@torch.no_grad()
def convert_diffusion_textual_inversion(
    context: ConversionContext,
    name: str,
    base_model: str,
    inversion: str,
    format: str,
    base_token: Optional[str] = None,
):
    dest_path = path.join(context.model_path, f"inversion-{name}")
    logger.info(
        "converting Textual Inversion: %s + %s -> %s", base_model, inversion, dest_path
    )

    encoder_path = path.join(dest_path, "text_encoder")
    encoder_model = path.join(encoder_path, "model.onnx")
    tokenizer_path = path.join(dest_path, "tokenizer")

    if (
        path.exists(dest_path)
        and path.exists(encoder_model)
        and path.exists(tokenizer_path)
    ):
        logger.info("ONNX model already exists, skipping.")
        return

    makedirs(encoder_path, exist_ok=True)

    if format == "concept":
        embeds_file = hf_hub_download(repo_id=inversion, filename="learned_embeds.bin")
        token_file = hf_hub_download(repo_id=inversion, filename="token_identifier.txt")

        with open(token_file, "r") as f:
            token = base_token or f.read()

        loaded_embeds = torch.load(embeds_file, map_location=context.map_location)

        # separate token and the embeds
        trained_token = list(loaded_embeds.keys())[0]
        embeds = loaded_embeds[trained_token]
    elif format == "embeddings":
        loaded_embeds = torch.load(inversion, map_location=context.map_location)

        string_to_token = loaded_embeds["string_to_token"]
        string_to_param = loaded_embeds["string_to_param"]

        # separate token and embeds
        trained_token = list(string_to_token.keys())[0]
        embeds = string_to_param[trained_token]

        num_tokens = embeds.shape[0]
        logger.info("generating %s layer tokens", num_tokens)
        token = [f"{base_token or name}-{i}" for i in range(num_tokens)]
    else:
        raise ValueError(f"unknown textual inversion format: {format}")

    logger.info("found embeddings for token %s: %s", token, embeds.shape)

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
    )

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    logger.info("added %s tokens", num_added_tokens)

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    if len(embeds.shape) == 2:
        # multiple vectors in embeds
        for i in range(embeds.shape[0]):
            layer_embeds = embeds[i]
            layer_token = token[i]
            logger.debug(
                "embedding %s vector for layer %s", layer_embeds.shape, layer_token
            )
            token_id = tokenizer.convert_tokens_to_ids(layer_token)
            text_encoder.get_input_embeddings().weight.data[token_id] = layer_embeds
    else:
        # get the id for the token and assign the embeds
        token_id = tokenizer.convert_tokens_to_ids(token)
        text_encoder.get_input_embeddings().weight.data[token_id] = embeds

    # conversion stuff
    text_input = tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    logger.info("saving tokenizer for textual inversion")
    tokenizer.save_pretrained(tokenizer_path)

    logger.info("saving text encoder for textual inversion")
    export(
        text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        (text_input.input_ids.to(dtype=torch.int32)),
        f=encoder_model,
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        do_constant_folding=True,
        opset_version=context.opset,
    )

    logger.info("textual inversion saved to %s", dest_path)
