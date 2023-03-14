from logging import getLogger
from os import path
from sys import argv
from typing import Dict, Literal

import numpy as np
import torch
from onnx import TensorProto, load, numpy_helper
from onnx.checker import check_model
from onnx.external_data_helper import (
    convert_model_to_external_data,
    write_external_data_tensors,
)
from safetensors.torch import load_file

from onnx_web.convert.utils import ConversionContext

logger = getLogger(__name__)


###
# everything in this file is still super experimental and may not produce valid ONNX models
###


def fix_initializer_name(key: str):
    # lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight
    # lora, unet, up_block.3.attentions.2.transformer_blocks.0.attn2.to_out.0
    return key.replace(".", "_")


def fix_node_name(key: str):
    fixed_name = fix_initializer_name(key.replace("/", "_"))
    if fixed_name[0] == "_":
        return fixed_name[1:]
    else:
        return fixed_name


def merge_lora(
    fixed_name: str,
    lora_names: str,
    dest_path: str,
    dest_type: Literal["text_encoder", "unet"],
):
    base_model = load(fixed_name)
    lora_models = [load_file(name) for name in lora_names.split(",")]
    lora_weights = np.ones((len(lora_models)))

    if dest_type == "text_encoder":
        lora_prefix = "lora_te_"
    else:
        lora_prefix = f"lora_{dest_type}_"

    blended: Dict[str, np.ndarray] = {}
    for lora_model, lora_weight in zip(lora_models, lora_weights):
        for key in lora_model.keys():
            if ".lora_down" in key and lora_prefix in key:
                base_key = key[: key.index(".lora_down")].replace(
                    lora_prefix, ""
                )

                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"
                logger.info("blending weights for keys: %s, %s, %s", key, up_key, alpha_key)

                down_weight = lora_model[key].to(dtype=torch.float32)
                up_weight = lora_model[up_key].to(dtype=torch.float32)

                dim = down_weight.size()[0]
                alpha = lora_model.get(alpha_key).numpy() or dim

                try:
                    if len(up_weight.size()) == 2:
                        pt_weight = up_weight @ down_weight
                        weight = (pt_weight.numpy() * (alpha / dim))
                    else:
                        pt_weight = (
                            (
                                up_weight.squeeze(3).squeeze(2)
                                @ down_weight.squeeze(3).squeeze(2)
                            )
                            .unsqueeze(2)
                            .unsqueeze(3)
                        )
                        weight = (alpha * pt_weight.numpy())

                    weight *= lora_weight
                    if base_key in blended:
                        blended[base_key] += weight
                    else:
                        blended[base_key] = weight

                except Exception:
                    logger.exception(
                        "error blending weights for key %s", base_key
                    )

    logger.info(
        "updating %s of %s nodes: %s",
        len(blended.keys()),
        len(base_model.graph.initializer),
        list(blended.keys())
    )

    fixed_initializer_names = [
        fix_initializer_name(node.name) for node in base_model.graph.initializer
    ]
    # logger.info("fixed initializer names: %s", fixed_initializer_names)

    fixed_node_names = [
        fix_node_name(node.name) for node in base_model.graph.node
    ]
    # logger.info("fixed node names: %s", fixed_node_names)


    for base_key, pt_weight in blended.items():
        bias_key = base_key + "_bias"
        matmul_key = base_key + "_MatMul"
        weight_key = base_key + "_weight"

        if weight_key in fixed_initializer_names:
            i = fixed_initializer_names.index(weight_key)
            node = base_model.graph.initializer[i]
            logger.info("found weight node: %s", node.name)

            base_weights = numpy_helper.to_array(node)
            logger.info("found blended weights for base: %s, %s", pt_weight.shape, base_weights.shape)

            updated_node = numpy_helper.from_array(base_weights + pt_weight, node.name)
            del base_model.graph.initializer[i]
            base_model.graph.initializer.insert(i, updated_node)
        elif matmul_key in fixed_node_names:
            logger.info("TODO: find MatMul node for %s", matmul_key)

            i = fixed_node_names.index(matmul_key)
            node = base_model.graph.node[i]
            logger.info("found matmul node: %s", node.name)

            # find that MatMul
            matmul_name = node.input[1]
            logger.info("matmul inputs: %s", node.input)

            bi = fixed_initializer_names.index(matmul_name)
            bias_node = base_model.graph.initializer[bi]
            base_weights = numpy_helper.to_array(bias_node)
            logger.info("found blended weights for matmul: %s, %s", pt_weight.shape, base_weights.shape)

            if pt_weight.shape != base_weights.shape:
                logger.info("transposing weights to make them fit")
                pt_weight = np.transpose(pt_weight)

            updated_node = numpy_helper.from_array(base_weights + pt_weight, bias_node.name)
            del base_model.graph.initializer[bi]
            base_model.graph.initializer.insert(bi, updated_node)
        elif bias_key in fixed_initializer_names:
            logger.info("TODO: find MatMul and blend bias for %s", bias_key)
            # find the nodes using this bias initializer
            fixed_nodes = [node for node in base_model.graph.node if fix_node_name(node.name).startswith(base_key)]
            logger.info("found %s nodes: %s", len(fixed_nodes), [node.name for node in fixed_nodes])

            if len(fixed_nodes) == 0:
                continue

            # find the MatMul in that node's inputs
            matmul_node = fixed_nodes[0]
            logger.info("matmul inputs: %s", matmul_node.input)
            # find that MatMul in the initializers
            matmul_name = matmul_node.input[1]
            i = fixed_initializer_names.index(matmul_name)
            node = base_model.graph.initializer[i]
            logger.info("found matmul node: %s", node.name)

            base_weights = numpy_helper.to_array(node)
            logger.info("found blended weights for bias: %s, %s", pt_weight.shape, base_weights.shape)

            if pt_weight.shape != base_weights.shape:
                logger.info("transposing weights to make them fit")
                pt_weight = np.transpose(pt_weight)

            bi = fixed_initializer_names.index(bias_key)
            bias_node = base_model.graph.initializer[bi]
            logger.info("bias shape: %s", numpy_helper.to_array(bias_node).shape)

            updated_node = numpy_helper.from_array(base_weights + pt_weight, node.name)
            del base_model.graph.initializer[i]
            base_model.graph.initializer.insert(i, updated_node)
        else:
            i = None
            node = None
            logger.info("could not find any nodes for %s", base_key)

    logger.info("sizes: %s -> %s, %s -> %s", len(fixed_initializer_names), len(base_model.graph.initializer), len(fixed_node_names), len(base_model.graph.node))

    # save it back to disk
    # TODO: save to memory instead
    convert_model_to_external_data(
        base_model,
        all_tensors_to_one_file=True,
        location=f"lora-{dest_type}-external.pb",
    )
    bare_model = write_external_data_tensors(base_model, dest_path)

    dest_file = path.join(dest_path, f"lora-{dest_type}.onnx")
    with open(dest_file, "wb") as model_file:
        model_file.write(bare_model.SerializeToString())

    logger.info("model saved, checking...")
    check_model(dest_file)

    logger.info("model successfully exported")


if __name__ == "__main__":
    merge_lora(*argv[1:])


