from logging import getLogger
from os import makedirs, path
from typing import List, Optional, Tuple

import numpy as np
import torch
from huggingface_hub.file_download import hf_hub_download
from onnx import ModelProto, load_model, numpy_helper, save_model
from transformers import CLIPTokenizer

from ...server.context import ServerContext
from ..utils import ConversionContext

logger = getLogger(__name__)


@torch.no_grad()
def blend_textual_inversions(
    context: ServerContext,
    text_encoder: ModelProto,
    tokenizer: CLIPTokenizer,
    inversions: List[Tuple[str, float, Optional[str], Optional[str]]],
) -> Tuple[ModelProto, CLIPTokenizer]:
    dtype = np.float
    embeds = {}

    for name, weight, base_token, format in inversions:
        if base_token is None:
            base_token = name

        if format is None:
            # TODO: detect concept format
            format = "embeddings"

        logger.info(
            "blending Textual Inversion %s with weight of %s for token %s",
            name,
            weight,
            base_token,
        )

        if format == "concept":
            # TODO: this should be done in fetch, maybe
            embeds_file = hf_hub_download(repo_id=name, filename="learned_embeds.bin")
            token_file = hf_hub_download(repo_id=name, filename="token_identifier.txt")

            with open(token_file, "r") as f:
                token = base_token or f.read()

            loaded_embeds = torch.load(embeds_file)

            # separate token and the embeds
            trained_token = list(loaded_embeds.keys())[0]

            layer = loaded_embeds[trained_token].cpu().numpy().astype(dtype)
            layer *= weight
            if trained_token in embeds:
                embeds[token] += layer
            else:
                embeds[token] = layer
        elif format == "embeddings":
            loaded_embeds = torch.load(name)

            string_to_token = loaded_embeds["string_to_token"]
            string_to_param = loaded_embeds["string_to_param"]

            # separate token and embeds
            trained_token = list(string_to_token.keys())[0]
            trained_embeds = string_to_param[trained_token]

            num_tokens = trained_embeds.shape[0]
            logger.debug("generating %s layer tokens for %s", num_tokens, name)

            sum_layer = np.zeros(trained_embeds[0, :].shape)

            for i in range(num_tokens):
                token = f"{base_token}-{i}"
                layer = trained_embeds[i, :].cpu().numpy().astype(dtype)
                layer *= weight

                sum_layer += layer
                if token in embeds:
                    embeds[token] += layer
                else:
                    embeds[token] = layer

            # add sum layer to embeds
            sum_token = f"{base_token}-all"
            if sum_token in embeds:
                embeds[sum_token] += sum_layer
            else:
                embeds[sum_token] = sum_layer
        else:
            raise ValueError(f"unknown Textual Inversion format: {format}")

        # add the tokens to the tokenizer
        logger.debug(
            "found embeddings for %s tokens: %s", len(embeds.keys()), embeds.keys()
        )
        num_added_tokens = tokenizer.add_tokens(list(embeds.keys()))
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
            )

        logger.trace("added %s tokens", num_added_tokens)

        # resize the token embeddings
        # text_encoder.resize_token_embeddings(len(tokenizer))
        embedding_node = [
            n
            for n in text_encoder.graph.initializer
            if n.name == "text_model.embeddings.token_embedding.weight"
        ][0]
        embedding_weights = numpy_helper.to_array(embedding_node)

        weights_dim = embedding_weights.shape[1]
        zero_weights = np.zeros((num_added_tokens, weights_dim))
        embedding_weights = np.concatenate((embedding_weights, zero_weights), axis=0)

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
                    embedding_weights.astype(np.float32), embedding_node.name
                )
                logger.trace("new initializer data type: %s", new_initializer.data_type)
                del text_encoder.graph.initializer[i]
                text_encoder.graph.initializer.insert(i, new_initializer)

    return (text_encoder, tokenizer)


@torch.no_grad()
def convert_diffusion_textual_inversion(
    context: ConversionContext,
    name: str,
    base_model: str,
    inversion: str,
    format: str,
    base_token: Optional[str] = None,
    weight: Optional[float] = 1.0,
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

    text_encoder = load_model(path.join(base_model, "text_encoder", "model.onnx"))
    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder, tokenizer = blend_textual_inversions(
        context,
        text_encoder,
        tokenizer,
        [(inversion, weight, base_token, format)],
    )

    logger.info("saving tokenizer for Textual Inversion")
    tokenizer.save_pretrained(tokenizer_path)

    logger.info("saving text encoder for Textual Inversion")
    save_model(
        text_encoder,
        f=encoder_model,
    )

    logger.info("Textual Inversion saved to %s", dest_path)
