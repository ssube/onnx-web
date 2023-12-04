from logging import getLogger
from os import makedirs, path
from typing import List, Optional, Tuple

import numpy as np
import torch
from onnx import ModelProto, load_model, numpy_helper, save_model
from transformers import CLIPTokenizer

from ...constants import ONNX_MODEL
from ...server.context import ServerContext
from ..utils import ConversionContext, load_tensor

logger = getLogger(__name__)


def detect_embedding_format(loaded_embeds) -> str:
    keys: List[str] = list(loaded_embeds.keys())
    if len(keys) == 1 and keys[0].startswith("<") and keys[0].endswith(">"):
        logger.debug("detected Textual Inversion concept: %s", keys)
        return "concept"
    elif "emb_params" in keys:
        logger.debug("detected Textual Inversion parameter embeddings: %s", keys)
        return "parameters"
    elif "string_to_token" in keys and "string_to_param" in keys:
        logger.debug("detected Textual Inversion token embeddings: %s", keys)
        return "embeddings"
    else:
        logger.error("unknown Textual Inversion format, no recognized keys: %s", keys)
        return None


def blend_embedding_concept(embeds, loaded_embeds, dtype, base_token, weight):
    # separate token and the embeds
    token = list(loaded_embeds.keys())[0]

    layer = loaded_embeds[token].numpy().astype(dtype)
    layer *= weight

    if base_token in embeds:
        embeds[base_token] += layer
    else:
        embeds[base_token] = layer

    if token in embeds:
        embeds[token] += layer
    else:
        embeds[token] = layer


def blend_embedding_parameters(embeds, loaded_embeds, dtype, base_token, weight):
    emb_params = loaded_embeds["emb_params"]

    num_tokens = emb_params.shape[0]
    logger.debug("generating %s layer tokens for %s", num_tokens, base_token)

    sum_layer = np.zeros(emb_params[0, :].shape)

    for i in range(num_tokens):
        token = f"{base_token}-{i}"
        layer = emb_params[i, :].numpy().astype(dtype)
        layer *= weight

        sum_layer += layer
        if token in embeds:
            embeds[token] += layer
        else:
            embeds[token] = layer

    # add base and sum tokens to embeds
    if base_token in embeds:
        embeds[base_token] += sum_layer
    else:
        embeds[base_token] = sum_layer

    sum_token = f"{base_token}-all"
    if sum_token in embeds:
        embeds[sum_token] += sum_layer
    else:
        embeds[sum_token] = sum_layer


def blend_embedding_embeddings(embeds, loaded_embeds, dtype, base_token, weight):
    string_to_token = loaded_embeds["string_to_token"]
    string_to_param = loaded_embeds["string_to_param"]

    # separate token and embeds
    token = list(string_to_token.keys())[0]
    trained_embeds = string_to_param[token]

    num_tokens = trained_embeds.shape[0]
    logger.debug("generating %s layer tokens for %s", num_tokens, base_token)

    sum_layer = np.zeros(trained_embeds[0, :].shape)

    for i in range(num_tokens):
        token = f"{base_token}-{i}"
        layer = trained_embeds[i, :].numpy().astype(dtype)
        layer *= weight

        sum_layer += layer
        if token in embeds:
            embeds[token] += layer
        else:
            embeds[token] = layer

    # add base and sum tokens to embeds
    if base_token in embeds:
        embeds[base_token] += sum_layer
    else:
        embeds[base_token] = sum_layer

    sum_token = f"{base_token}-all"
    if sum_token in embeds:
        embeds[sum_token] += sum_layer
    else:
        embeds[sum_token] = sum_layer


def blend_embedding_node(text_encoder, tokenizer, embeds, num_added_tokens):
    # resize the token embeddings
    # text_encoder.resize_token_embeddings(len(tokenizer))
    embedding_node = [
        n
        for n in text_encoder.graph.initializer
        if n.name == "text_model.embeddings.token_embedding.weight"
    ][0]
    base_weights = numpy_helper.to_array(embedding_node)

    weights_dim = base_weights.shape[1]
    zero_weights = np.zeros((num_added_tokens, weights_dim))
    embedding_weights = np.concatenate((base_weights, zero_weights), axis=0)

    for token, weights in embeds.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.trace("embedding %s weights for token %s", weights.shape, token)
        embedding_weights[token_id] = weights

    # replace embedding_node
    for i in range(len(text_encoder.graph.initializer)):
        if (
            text_encoder.graph.initializer[i].name
            == "text_model.embeddings.token_embedding.weight"
        ):
            new_initializer = numpy_helper.from_array(
                embedding_weights.astype(base_weights.dtype), embedding_node.name
            )
            logger.trace("new initializer data type: %s", new_initializer.data_type)
            del text_encoder.graph.initializer[i]
            text_encoder.graph.initializer.insert(i, new_initializer)


@torch.no_grad()
def blend_textual_inversions(
    server: ServerContext,
    text_encoder: ModelProto,
    tokenizer: CLIPTokenizer,
    embeddings: List[Tuple[str, float, Optional[str], Optional[str]]],
) -> Tuple[ModelProto, CLIPTokenizer]:
    # always load to CPU for blending
    device = torch.device("cpu")
    dtype = np.float32
    embeds = {}

    for name, weight, base_token, format in embeddings:
        if base_token is None:
            logger.debug("no base token provided, using name: %s", name)
            base_token = name

        logger.info(
            "blending Textual Inversion %s with weight of %s for token %s",
            name,
            weight,
            base_token,
        )

        loaded_embeds = load_tensor(name, map_location=device)
        if loaded_embeds is None:
            logger.warning("unable to load tensor")
            continue

        if format is None:
            format = detect_embedding_format(loaded_embeds)

        if format == "concept":
            blend_embedding_concept(embeds, loaded_embeds, dtype, base_token, weight)
        elif format == "parameters":
            blend_embedding_parameters(embeds, loaded_embeds, dtype, base_token, weight)
        elif format == "embeddings":
            blend_embedding_embeddings(embeds, loaded_embeds, dtype, base_token, weight)
        else:
            raise ValueError(f"unknown Textual Inversion format: {format}")

    # add the tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(list(embeds.keys()))
    if num_added_tokens == 0:
        raise ValueError(
            "The tokenizer already contains the tokens. Please pass a different `token` that is not already in the tokenizer."
        )

    logger.trace("added %s tokens", num_added_tokens)

    blend_embedding_node(text_encoder, tokenizer, embeds, num_added_tokens)

    return (text_encoder, tokenizer)


@torch.no_grad()
def convert_diffusion_textual_inversion(
    conversion: ConversionContext,
    name: str,
    base_model: str,
    inversion: str,
    inversion_format: str,
    base_token: Optional[str] = None,
    inversion_weight: Optional[float] = 1.0,
):
    dest_path = path.join(conversion.model_path, f"inversion-{name}")
    logger.info(
        "converting Textual Inversion: %s + %s -> %s", base_model, inversion, dest_path
    )

    encoder_path = path.join(dest_path, "text_encoder")
    encoder_model = path.join(encoder_path, ONNX_MODEL)
    tokenizer_path = path.join(dest_path, "tokenizer")

    if (
        path.exists(dest_path)
        and path.exists(encoder_model)
        and path.exists(tokenizer_path)
    ):
        logger.info("ONNX model already exists, skipping.")
        return

    makedirs(encoder_path, exist_ok=True)

    text_encoder = load_model(path.join(base_model, "text_encoder", ONNX_MODEL))
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder, tokenizer = blend_textual_inversions(
        conversion,
        text_encoder,
        tokenizer,
        [(inversion, inversion_weight, base_token, inversion_format)],
    )

    logger.info("saving tokenizer for textual inversion")
    tokenizer.save_pretrained(tokenizer_path)

    logger.info("saving text encoder for textual inversion")
    save_model(
        text_encoder,
        f=encoder_model,
    )

    logger.info("textual inversion saved to %s", dest_path)
