from itertools import groupby
from logging import getLogger
from os import path
from sys import argv
from typing import List, Literal, Tuple

import torch
from onnx import TensorProto, load, numpy_helper
from onnx.checker import check_model
from onnx.external_data_helper import convert_model_to_external_data, write_external_data_tensors
from safetensors.torch import load_file

# from ..utils import ConversionContext

logger = getLogger(__name__)


###
# everything in this file is still super experimental and may not produce valid ONNX models
###


def fix_name(key: str):
    # lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight
    # lora, unet, up_block.3.attentions.2.transformer_blocks.0.attn2.to_out.0
    return key.replace(".", "_")


def merge_lora(base_name: str, lora_names: str, dest_path: str, dest_type: Literal["text_encoder", "unet"]):
    base_model = load(base_name)
    lora_models = [load_file(name) for name in lora_names.split(",")]

    lora_nodes: List[Tuple[int, TensorProto]] = []

    fixed_initialized_names = [fix_name(node.name) for node in base_model.graph.initializer]
    logger.info("fixed initializer names: %s", fixed_initialized_names)

    if dest_type == "text_encoder":
        lora_prefix = "lora_te_"
    elif dest_type == "unet":
        lora_prefix = "lora_unet_"
    else:
        lora_prefix = "lora_"

    for i in range(len(fixed_initialized_names)):
        base_key = fixed_initialized_names[i]
        base_node = base_model.graph.initializer[i]

        updates = []
        for lora_model in lora_models:
            for key in lora_model.keys():
                if ".lora_down" in key:
                    original_key = key[: key.index(".lora_down")].replace(lora_prefix, "")
                    bias_key = original_key + "_bias"
                    weight_key = original_key + "_weight"

                    if bias_key.startswith(base_key):
                        print("found bias key:", base_key, bias_key)

                    if weight_key == base_key:
                        print("down for key:", base_key, weight_key)

                        up_key = key.replace("lora_down", "lora_up")
                        alpha_key = key[: key.index("lora_down")] + "alpha"

                        down_weight = lora_model[key].to(dtype=torch.float32)
                        up_weight = lora_model[up_key].to(dtype=torch.float32)

                        dim = down_weight.size()[0]
                        alpha = lora_model.get(alpha_key).numpy() or dim

                        np_vals = numpy_helper.to_array(base_node)
                        print("before shape", np_vals.shape, up_weight.shape, down_weight.shape)

                        try:
                            if len(up_weight.size()) == 2:
                                squoze = up_weight @ down_weight
                                print(squoze.shape)
                                np_vals = np_vals + (squoze.numpy() * (alpha / dim))
                            else:
                                squoze = (
                                    (
                                        up_weight.squeeze(3).squeeze(2)
                                        @ down_weight.squeeze(3).squeeze(2)
                                    )
                                    .unsqueeze(2)
                                    .unsqueeze(3)
                                )
                                print(squoze.shape)
                                np_vals = np_vals + (alpha * squoze.numpy())
                                print("after shape", np_vals.shape)

                                updates.append(np_vals)

                            break
                        except Exception as e:
                            logger.exception("error blending weights with key %s", weight_key)

        if len(updates) == 0:
            logger.debug("no lora found for key %s", base_key)
        else:
            # blend updates together and append to lora_nodes
            logger.info("blending %s updated weights for key %s", len(updates), base_key)

            # TODO: allow individual alphas
            np_vals = sum(updates) / len(updates)

            retensor = numpy_helper.from_array(np_vals, base_node.name)
            logger.info("created new tensor with %s bytes", len(retensor.raw_data))

            # TypeError: does not support assignment
            lora_nodes.append((i, retensor))


    logger.info("updating %s of %s nodes", len(lora_nodes), len(base_model.graph.initializer))
    for idx, node in lora_nodes:
        del base_model.graph.initializer[idx]
        base_model.graph.initializer.insert(idx, node)

    # save it back to disk
    # TODO: save to memory instead
    convert_model_to_external_data(base_model, all_tensors_to_one_file=True, location=f"lora-{dest_type}-external.pb")
    bare_model = write_external_data_tensors(base_model, dest_path)

    dest_file = path.join(dest_path, f"lora-{dest_type}.onnx")
    with open(dest_file, "wb") as model_file:
        model_file.write(bare_model.SerializeToString())

    logger.info("model saved, checking...")
    check_model(dest_file)

    logger.info("model successfully exported")


if __name__ == "__main__":
    merge_lora(*argv[1:])
