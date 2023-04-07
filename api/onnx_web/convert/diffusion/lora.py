from argparse import ArgumentParser
from logging import getLogger
from os import path
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import torch
from onnx import ModelProto, load, numpy_helper
from onnx.checker import check_model
from onnx.external_data_helper import (
    convert_model_to_external_data,
    set_external_data,
    write_external_data_tensors,
)
from onnxruntime import InferenceSession, OrtValue, SessionOptions

from ...server.context import ServerContext
from ..utils import ConversionContext, load_tensor

logger = getLogger(__name__)


def buffer_external_data_tensors(
    model: ModelProto,
) -> Tuple[ModelProto, List[Tuple[str, OrtValue]]]:
    external_data = []
    for tensor in model.graph.initializer:
        name = tensor.name

        logger.trace("externalizing tensor: %s", name)
        if tensor.HasField("raw_data"):
            npt = numpy_helper.to_array(tensor)
            orv = OrtValue.ortvalue_from_numpy(npt)
            external_data.append((name, orv))
            # mimic set_external_data
            set_external_data(tensor, location="foo.bin")
            tensor.name = name
            tensor.ClearField("raw_data")

    return (model, external_data)


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


def blend_loras(
    _context: ServerContext,
    base_name: Union[str, ModelProto],
    loras: List[Tuple[str, float]],
    model_type: Literal["text_encoder", "unet"],
):
    # always load to CPU for blending
    device = torch.device("cpu")
    dtype = torch.float32

    base_model = base_name if isinstance(base_name, ModelProto) else load(base_name)
    lora_models = [load_tensor(name, map_location=device) for name, _weight in loras]

    if model_type == "text_encoder":
        lora_prefix = "lora_te_"
    else:
        lora_prefix = f"lora_{model_type}_"

    blended: Dict[str, np.ndarray] = {}
    for (lora_name, lora_weight), lora_model in zip(loras, lora_models):
        logger.debug("blending LoRA from %s with weight of %s", lora_name, lora_weight)
        if lora_model is None:
            logger.warning("unable to load tensor for LoRA")
            continue

        for key in lora_model.keys():
            if ".lora_down" in key and lora_prefix in key:
                base_key = key[: key.index(".lora_down")].replace(lora_prefix, "")

                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"
                logger.trace(
                    "blending weights for keys: %s, %s, %s", key, up_key, alpha_key
                )

                down_weight = lora_model[key].to(dtype=dtype)
                up_weight = lora_model[up_key].to(dtype=dtype)

                dim = down_weight.size()[0]
                alpha = lora_model.get(alpha_key, dim).to(dtype).numpy()

                try:
                    if len(down_weight.size()) == 2:
                        # blend for nn.Linear
                        logger.trace(
                            "blending weights for Linear node: %s, %s, %s",
                            down_weight.shape,
                            up_weight.shape,
                            alpha,
                        )
                        weights = up_weight @ down_weight
                        np_weights = weights.numpy() * (alpha / dim)
                    elif len(down_weight.size()) == 4 and down_weight.shape[-2:] == (
                        1,
                        1,
                    ):
                        # blend for nn.Conv2d 1x1
                        logger.trace(
                            "blending weights for Conv 1x1 node: %s, %s, %s",
                            down_weight.shape,
                            up_weight.shape,
                            alpha,
                        )
                        weights = (
                            (
                                up_weight.squeeze(3).squeeze(2)
                                @ down_weight.squeeze(3).squeeze(2)
                            )
                            .unsqueeze(2)
                            .unsqueeze(3)
                        )
                        np_weights = weights.numpy() * (alpha / dim)
                    elif len(down_weight.size()) == 4 and down_weight.shape[-2:] == (
                        3,
                        3,
                    ):
                        # blend for nn.Conv2d 3x3
                        logger.trace(
                            "blending weights for Conv 3x3 node: %s, %s, %s",
                            down_weight.shape,
                            up_weight.shape,
                            alpha,
                        )
                        weights = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3)
                        np_weights = weights.numpy() * (alpha / dim)
                    else:
                        logger.warning(
                            "unknown LoRA node type at %s: %s",
                            base_key,
                            up_weight.shape[-2:],
                        )
                        continue

                    np_weights *= lora_weight
                    if base_key in blended:
                        blended[base_key] += np_weights
                    else:
                        blended[base_key] = np_weights

                except Exception:
                    logger.exception("error blending weights for key %s", base_key)

    logger.trace(
        "updating %s of %s initializers: %s",
        len(blended.keys()),
        len(base_model.graph.initializer),
        list(blended.keys()),
    )

    fixed_initializer_names = [
        fix_initializer_name(node.name) for node in base_model.graph.initializer
    ]
    logger.trace("fixed initializer names: %s", fixed_initializer_names)

    fixed_node_names = [fix_node_name(node.name) for node in base_model.graph.node]
    logger.trace("fixed node names: %s", fixed_node_names)

    for base_key, weights in blended.items():
        conv_key = base_key + "_Conv"
        matmul_key = base_key + "_MatMul"

        logger.trace(
            "key %s has conv: %s, matmul: %s",
            base_key,
            conv_key in fixed_node_names,
            matmul_key in fixed_node_names,
        )

        if conv_key in fixed_node_names:
            conv_idx = fixed_node_names.index(conv_key)
            conv_node = base_model.graph.node[conv_idx]
            logger.trace("found conv node %s using %s", conv_node.name, conv_node.input)

            # find weight initializer
            weight_name = [n for n in conv_node.input if ".weight" in n][0]
            weight_name = fix_initializer_name(weight_name)

            weight_idx = fixed_initializer_names.index(weight_name)
            weight_node = base_model.graph.initializer[weight_idx]
            logger.trace("found weight initializer: %s", weight_node.name)

            # blending
            base_weights = numpy_helper.to_array(weight_node)
            logger.trace(
                "found blended weights for conv: %s, %s",
                weights.shape,
                base_weights.shape,
            )

            if base_weights.shape[-2:] == (1, 1):
                blended = base_weights.squeeze((3, 2)) + weights.squeeze((3, 2))
                blended = np.expand_dims(blended, (2, 3))
            else:
                blended = base_weights + weights

            logger.trace("blended weight shape: %s", blended.shape)

            # replace the original initializer
            updated_node = numpy_helper.from_array(
                blended.astype(base_weights.dtype), weight_node.name
            )
            del base_model.graph.initializer[weight_idx]
            base_model.graph.initializer.insert(weight_idx, updated_node)
        elif matmul_key in fixed_node_names:
            weight_idx = fixed_node_names.index(matmul_key)
            weight_node = base_model.graph.node[weight_idx]
            logger.trace(
                "found matmul node %s using %s", weight_node.name, weight_node.input
            )

            # find the MatMul initializer
            matmul_name = [n for n in weight_node.input if "MatMul" in n][0]

            matmul_idx = fixed_initializer_names.index(matmul_name)
            matmul_node = base_model.graph.initializer[matmul_idx]
            logger.trace("found matmul initializer: %s", matmul_node.name)

            # blending
            base_weights = numpy_helper.to_array(matmul_node)
            logger.trace(
                "found blended weights for matmul: %s, %s",
                weights.shape,
                base_weights.shape,
            )

            blended = base_weights + weights.transpose()
            logger.trace("blended weight shape: %s", blended.shape)

            # replace the original initializer
            updated_node = numpy_helper.from_array(
                blended.astype(base_weights.dtype), matmul_node.name
            )
            del base_model.graph.initializer[matmul_idx]
            base_model.graph.initializer.insert(matmul_idx, updated_node)
        else:
            logger.warning("could not find any nodes for %s", base_key)

    logger.debug(
        "node counts: %s -> %s, %s -> %s",
        len(fixed_initializer_names),
        len(base_model.graph.initializer),
        len(fixed_node_names),
        len(base_model.graph.node),
    )

    return base_model


if __name__ == "__main__":
    context = ConversionContext.from_environ()
    parser = ArgumentParser()
    parser.add_argument("--base", type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--type", type=str, choices=["text_encoder", "unet"])
    parser.add_argument("--lora_models", nargs="+", type=str, default=[])
    parser.add_argument("--lora_weights", nargs="+", type=float, default=[])

    args = parser.parse_args()
    logger.info(
        "merging %s with %s with weights: %s",
        args.lora_models,
        args.base,
        args.lora_weights,
    )

    default_weight = 1.0 / len(args.lora_models)
    while len(args.lora_weights) < len(args.lora_models):
        args.lora_weights.append(default_weight)

    blend_model = blend_loras(
        context,
        args.base,
        list(zip(args.lora_models, args.lora_weights)),
        args.type,
    )
    if args.dest is None or args.dest == "" or args.dest == ":load":
        # convert to external data and save to memory
        (bare_model, external_data) = buffer_external_data_tensors(blend_model)
        logger.info("saved external data for %s nodes", len(external_data))

        external_names, external_values = zip(*external_data)
        opts = SessionOptions()
        opts.add_external_initializers(list(external_names), list(external_values))
        sess = InferenceSession(
            bare_model.SerializeToString(),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        logger.info(
            "successfully loaded blended model: %s", [i.name for i in sess.get_inputs()]
        )
    else:
        convert_model_to_external_data(
            blend_model, all_tensors_to_one_file=True, location=f"lora-{args.type}.pb"
        )
        bare_model = write_external_data_tensors(blend_model, args.dest)
        dest_file = path.join(args.dest, f"lora-{args.type}.onnx")

        with open(dest_file, "w+b") as model_file:
            model_file.write(bare_model.SerializeToString())

        logger.info("successfully saved blended model: %s", dest_file)

        check_model(dest_file)

        logger.info("checked blended model")
