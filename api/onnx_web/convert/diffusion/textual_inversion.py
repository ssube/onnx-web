from os import mkdirs, path
from huggingface_hub.file_download import hf_hub_download
from transformers import CLIPTokenizer, CLIPTextModel
from torch.onnx import export
from logging import getLogger

from ..utils import ConversionContext

import torch

logger = getLogger(__name__)


def convert_diffusion_textual_inversion(context: ConversionContext, name: str, base_model: str, inversion: str):
    dest_path = path.join(context.model_path, f"inversion-{name}")
    logger.info("converting Textual Inversion: %s + %s -> %s", base_model, inversion, dest_path)

    if path.exists(dest_path):
        logger.info("ONNX model already exists, skipping.")

    mkdirs(path.join(dest_path, "text_encoder"))

    embeds_file = hf_hub_download(repo_id=inversion, filename="learned_embeds.bin")
    token_file = hf_hub_download(repo_id=inversion, filename="token_identifier.txt")

    with open(token_file, "r") as f:
        token = f.read()

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model,
        subfolder="text_encoder",
    )

    loaded_embeds = torch.load(embeds_file, map_location=context.map_location)

    # separate token and the embeds
    trained_token = list(loaded_embeds.keys())[0]
    embeds = loaded_embeds[trained_token]

    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)

    # add the token in tokenizer
    num_added_tokens = tokenizer.add_tokens(token)
    if num_added_tokens == 0:
        raise ValueError(
            f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer."
        )

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

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

    export(
        text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        (
            text_input.input_ids.to(dtype=torch.int32)
        ),
        f=path.join(dest_path, "text_encoder", "model.onnx"),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        do_constant_folding=True,
        opset_version=context.opset,
    )
