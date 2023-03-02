from logging import getLogger
from os import path
from sys import argv
from typing import List, Tuple

import onnx.checker
import torch
from numpy import ndarray
from onnx import ModelProto, TensorProto, helper, load, numpy_helper, save_model
from safetensors import safe_open

from ..utils import ConversionContext

logger = getLogger(__name__)


###
# everything in this file is still super experimental and may not produce valid ONNX models
###


def load_lora(filename: str):
    model = load(filename)

    for weight in model.graph.initializer:
        # print(weight.name, numpy_helper.to_array(weight).shape)
        pass

    return model


def blend_loras(
    base: ModelProto, weights: List[ModelProto], alphas: List[float]
) -> List[Tuple[TensorProto, ndarray]]:
    total = 1 + sum(alphas)

    results = []

    for base_node in base.graph.initializer:
        logger.info("blending initializer node %s", base_node.name)
        base_weights = numpy_helper.to_array(base_node).copy()

        for weight, alpha in zip(weights, alphas):
            weight_node = next(
                iter([f for f in weight.graph.initializer if f.name == base_node.name]),
                None,
            )

            if weight_node is not None:
                base_weights += numpy_helper.to_array(weight_node) * alpha
            else:
                logger.warning(
                    "missing weights: %s in %s", base_node.name, weight.doc_string
                )

        results.append((base_node, base_weights / total))

    return results


def convert_diffusion_lora(context: ConversionContext, component: str):
    lora_weights = [
        f"diffusion-lora-jack/{component}/model.onnx",
        f"diffusion-lora-taters/{component}/model.onnx",
    ]

    base = load_lora(f"stable-diffusion-onnx-v1-5/{component}/model.onnx")
    weights = [load_lora(f) for f in lora_weights]
    alphas = [1 / len(weights)] * len(weights)
    logger.info("blending LoRAs with alphas: %s, %s", weights, alphas)

    result = blend_loras(base, weights, alphas)
    logger.info("blended result keys: %s", len(result))

    del weights
    del alphas

    tensors = []
    for node, tensor in result:
        logger.info("remaking tensor for %s", node.name)
        tensors.append(helper.make_tensor(node.name, node.data_type, node.dims, tensor))

    del result

    graph = helper.make_graph(
        base.graph.node,
        base.graph.name,
        base.graph.input,
        base.graph.output,
        tensors,
        base.graph.doc_string,
        base.graph.value_info,
        base.graph.sparse_initializer,
    )
    model = helper.make_model(graph)

    del model.opset_import[:]
    opset = model.opset_import.add()
    opset.version = 14

    onnx_path = path.join(context.cache_path, f"lora-{component}.onnx")
    tensor_path = path.join(context.cache_path, f"lora-{component}.tensors")
    save_model(
        model,
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=tensor_path,
    )
    logger.info(
        "saved model to %s and tensors to %s",
        onnx_path,
        tensor_path,
    )


def fix_key(key: str):
    # lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight
    # lora, unet, up_block.3.attentions.2.transformer_blocks.0.attn2.to_out.0
    return key.replace(".", "_")


def merge_lora():
    base_name = argv[1]
    lora_name = argv[2]

    base_model = load(base_name)
    lora_model = safe_open(lora_name, framework="pt")

    lora_nodes = []
    for base_node in base_model.graph.initializer:
        base_key = fix_key(base_node.name)

        for key in lora_model.keys():
            if "lora_down" in key:
                lora_key = key[: key.index("lora_down")].replace("lora_unet_", "")
                if lora_key.startswith(base_key):
                    print("down for key:", base_key, lora_key)

                    up_key = key.replace("lora_down", "lora_up")
                    alpha_key = key[: key.index("lora_down")] + "alpha"

                    down_weight = lora_model.get_tensor(key).to(dtype=torch.float32)
                    up_weight = lora_model.get_tensor(up_key).to(dtype=torch.float32)

                    dim = down_weight.size()[0]
                    alpha = lora_model.get(alpha_key).numpy() or dim

                    np_vals = numpy_helper.to_array(base_node)
                    print(np_vals.shape, up_weight.shape, down_weight.shape)

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

                        # retensor = numpy_helper.from_array(np_vals, base_node.name)
                        retensor = helper.make_tensor(
                            base_node.name,
                            base_node.data_type,
                            base_node.dim,
                            np_vals,
                            raw=True,
                        )
                        print(retensor)

                        # TypeError: does not support assignment
                        lora_nodes.append(retensor)

                        break
                    except Exception as e:
                        print(e)

        if retensor is None:
            print("no lora found for key", base_key)
            lora_nodes.append(base_node)

    print(len(lora_nodes), len(base_model.graph.initializer))
    del base_model.graph.initializer[:]
    base_model.graph.initializer.extend(lora_nodes)

    onnx.checker.check_model(base_model)


if __name__ == "__main__":
    context = ConversionContext.from_environ()
    convert_diffusion_lora(context, "unet")
    convert_diffusion_lora(context, "text_encoder")
