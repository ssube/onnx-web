from logging import getLogger
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from onnx import ModelProto, NodeProto, TensorProto, load, numpy_helper
from onnx.external_data_helper import set_external_data
from onnxruntime import OrtValue
from scipy import interpolate

from ...server.context import ServerContext
from ..utils import load_tensor

logger = getLogger(__name__)


def sum_weights(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    logger.trace("summing weights with shapes: %s + %s", a.shape, b.shape)

    # if they are the same, simply add them
    if len(a.shape) == len(b.shape):
        return a + b

    # get the kernel size from the tensor with the higher rank
    if len(a.shape) > len(b.shape):
        kernel = a.shape[-2:]
        hr = a
        lr = b
    else:
        kernel = b.shape[-2:]
        hr = b
        lr = a

    if kernel == (1, 1):
        lr = np.expand_dims(lr, axis=(2, 3))  # TODO: generate axis

    return hr + lr


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


def fix_xl_names(keys: Dict[str, Any], nodes: List[NodeProto]) -> Dict[str, Any]:
    fixed = {}
    remaining = list(nodes)

    for key, value in keys.items():
        root, *rest = key.split(".")
        logger.trace("fixing XL node name: %s -> %s", key, root)

        simple = False
        if root.startswith("input"):
            block = "down_blocks"
        elif root.startswith("middle"):
            block = "mid_block"  # not plural
        elif root.startswith("output"):
            block = "up_blocks"
        elif root.startswith("text_model"):
            block = "text_model"
        elif root.startswith("down_blocks"):
            block = "down_blocks"
            simple = True
        elif root.startswith("mid_block"):
            block = "mid_block"
            simple = True
        elif root.startswith("up_blocks"):
            block = "up_blocks"
            simple = True
        else:
            logger.warning("unknown XL key name: %s", key)
            fixed[key] = value
            continue

        suffix = None
        for s in [
            "conv",
            "conv_shortcut",
            "conv1",
            "conv2",
            "fc1",
            "fc2",
            "ff_net_0_proj",
            "ff_net_2",
            "proj",
            "proj_in",
            "proj_out",
            "to_k",
            "to_out_0",
            "to_q",
            "to_v",
        ]:
            if root.endswith(s):
                suffix = s

        if suffix is None:
            logger.warning("new XL key type: %s", root)
            continue

        logger.trace("searching for XL node: %s -> /%s/*/%s", root, block, suffix)
        match: Optional[NodeProto] = None
        if "conv" in suffix:
            match = next(
                node for node in remaining if fix_node_name(node.name) == f"{root}_Conv"
            )
        elif "time_emb_proj" in root:
            match = next(
                node for node in remaining if fix_node_name(node.name) == f"{root}_Gemm"
            )
        elif block == "text_model" or simple:
            match = next(
                node
                for node in remaining
                if fix_node_name(node.name) == f"{root}_MatMul"
            )
        else:
            # search in order. one side has sparse indices, so they will not match.
            match = next(
                node
                for node in remaining
                if node.name.startswith(f"/{block}")
                and fix_node_name(node.name).endswith(
                    f"{suffix}_MatMul"
                )  # needs to be fixed because some places use to_out.0
            )

        if match is None:
            logger.warning("no matches for XL key: %s", root)
            continue
        else:
            logger.trace("matched key: %s -> %s", key, match.name)

        name: str = match.name
        name = fix_node_name(name)
        if name.endswith("_MatMul"):
            name = name[:-7]
        elif name.endswith("_Gemm"):
            name = name[:-5]
        elif name.endswith("_Conv"):
            name = name[:-5]

        logger.trace("matching XL key with node: %s -> %s, %s", key, match.name, name)

        fixed[name] = value
        remaining.remove(match)

    logger.debug(
        "SDXL LoRA key fixup matched %s keys, %s remaining",
        len(fixed.keys()),
        len(remaining),
    )

    return fixed


def kernel_slice(x: int, y: int, shape: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (
        min(x, shape[2] - 1),
        min(y, shape[3] - 1),
    )


def blend_weights_loha(
    key: str, lora_prefix: str, lora_model: Dict, dtype
) -> Tuple[str, np.ndarray]:
    base_key = key[: key.index(".hada_w1_a")].replace(lora_prefix, "")

    t1_key = key.replace("hada_w1_a", "hada_t1")
    t2_key = key.replace("hada_w1_a", "hada_t2")
    w1b_key = key.replace("hada_w1_a", "hada_w1_b")
    w2a_key = key.replace("hada_w1_a", "hada_w2_a")
    w2b_key = key.replace("hada_w1_a", "hada_w2_b")
    alpha_key = key[: key.index("hada_w1_a")] + "alpha"
    logger.trace(
        "blending weights for LoHA keys: %s, %s, %s, %s, %s",
        key,
        w1b_key,
        w2a_key,
        w2b_key,
        alpha_key,
    )

    w1a_weight = lora_model[key].to(dtype=dtype)
    w1b_weight = lora_model[w1b_key].to(dtype=dtype)
    w2a_weight = lora_model[w2a_key].to(dtype=dtype)
    w2b_weight = lora_model[w2b_key].to(dtype=dtype)

    t1_weight = lora_model.get(t1_key, None)
    t2_weight = lora_model.get(t2_key, None)

    dim = w1b_weight.size()[0]
    alpha = lora_model.get(alpha_key, dim).to(dtype).numpy()

    if t1_weight is not None and t2_weight is not None:
        t1_weight = t1_weight.to(dtype=dtype)
        t2_weight = t2_weight.to(dtype=dtype)

        logger.trace(
            "composing weights for LoHA node: (%s, %s, %s) * (%s, %s, %s)",
            t1_weight.shape,
            w1a_weight.shape,
            w1b_weight.shape,
            t2_weight.shape,
            w2a_weight.shape,
            w2b_weight.shape,
        )
        weights_1 = torch.einsum(
            "i j k l, j r, i p -> p r k l",
            t1_weight,
            w1b_weight,
            w1a_weight,
        )
        weights_2 = torch.einsum(
            "i j k l, j r, i p -> p r k l",
            t2_weight,
            w2b_weight,
            w2a_weight,
        )
        weights = weights_1 * weights_2
        np_weights = weights.numpy() * (alpha / dim)
    else:
        logger.trace(
            "blending weights for LoHA node: (%s @ %s) * (%s @ %s)",
            w1a_weight.shape,
            w1b_weight.shape,
            w2a_weight.shape,
            w2b_weight.shape,
        )
        weights = (w1a_weight @ w1b_weight) * (w2a_weight @ w2b_weight)
        np_weights = weights.numpy() * (alpha / dim)

    return base_key, np_weights


def blend_weights_lora(
    key: str, lora_prefix: str, lora_model: Dict, dtype
) -> Tuple[str, np.ndarray]:
    base_key = key[: key.index(".lora_down")].replace(lora_prefix, "")

    mid_key = key.replace("lora_down", "lora_mid")
    up_key = key.replace("lora_down", "lora_up")
    alpha_key = key[: key.index("lora_down")] + "alpha"
    logger.trace("blending weights for LoRA keys: %s, %s, %s", key, up_key, alpha_key)

    down_weight = lora_model[key].to(dtype=dtype)
    up_weight = lora_model[up_key].to(dtype=dtype)

    mid_weight = None
    if mid_key in lora_model:
        mid_weight = lora_model[mid_key].to(dtype=dtype)

    dim = down_weight.size()[0]
    alpha = lora_model.get(alpha_key, dim)

    if not isinstance(alpha, int):
        alpha = alpha.to(dtype).numpy()

    kernel = down_weight.shape[-2:]
    if mid_weight is not None:
        kernel = mid_weight.shape[-2:]

    if len(down_weight.size()) == 2:
        # blend for nn.Linear
        logger.trace(
            "blending weights for Linear node: (%s @ %s) * %s",
            down_weight.shape,
            up_weight.shape,
            alpha,
        )
        weights = up_weight @ down_weight
        np_weights = weights.numpy() * (alpha / dim)
    elif len(down_weight.size()) == 4 and kernel == (
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
            (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
            .unsqueeze(2)
            .unsqueeze(3)
        )
        np_weights = weights.numpy() * (alpha / dim)
    elif len(down_weight.size()) == 4 and kernel == (
        3,
        3,
    ):
        if mid_weight is not None:
            # blend for nn.Conv2d 3x3 with CP decomp
            logger.trace(
                "composing weights for Conv 3x3 node: %s, %s, %s, %s",
                down_weight.shape,
                up_weight.shape,
                mid_weight.shape,
                alpha,
            )
            weights = torch.zeros((up_weight.shape[0], down_weight.shape[1], *kernel))

            for w in range(kernel[0]):
                for h in range(kernel[1]):
                    weights[:, :, w, h] = (
                        up_weight.squeeze(3).squeeze(2) @ mid_weight[:, :, w, h]
                    ) @ down_weight.squeeze(3).squeeze(2)

            np_weights = weights.numpy() * (alpha / dim)
        else:
            # blend for nn.Conv2d 3x3
            logger.trace(
                "blending weights for Conv 3x3 node: %s, %s, %s",
                down_weight.shape,
                up_weight.shape,
                alpha,
            )
            weights = torch.zeros((up_weight.shape[0], down_weight.shape[1], *kernel))

            for w in range(kernel[0]):
                for h in range(kernel[1]):
                    down_w, down_h = kernel_slice(w, h, down_weight.shape)
                    up_w, up_h = kernel_slice(w, h, up_weight.shape)

                    weights[:, :, w, h] = (
                        up_weight[:, :, up_w, up_h] @ down_weight[:, :, down_w, down_h]
                    )

            np_weights = weights.numpy() * (alpha / dim)
    else:
        logger.warning(
            "unknown LoRA node type at %s: %s",
            base_key,
            up_weight.shape[-2:],
        )
        # TODO: should this be None?
        np_weights = np.zeros((1, 1, 1, 1))

    return base_key, np_weights


def blend_node_conv_gemm(weight_node, weights) -> TensorProto:
    # blending
    onnx_weights = numpy_helper.to_array(weight_node)
    logger.trace(
        "found blended weights for conv: %s, %s",
        onnx_weights.shape,
        weights.shape,
    )

    if onnx_weights.shape[-2:] == (1, 1):
        if weights.shape[-2:] == (1, 1):
            blended = onnx_weights.squeeze((3, 2)) + weights.squeeze((3, 2))
        else:
            blended = onnx_weights.squeeze((3, 2)) + weights

        blended = np.expand_dims(blended, (2, 3))
    else:
        if onnx_weights.shape != weights.shape:
            logger.warning(
                "reshaping weights for mismatched Conv node: %s, %s",
                onnx_weights.shape,
                weights.shape,
            )
            # TODO: test if this can be replaced with interpolation, simply reshaping is pretty sus
            blended = onnx_weights + weights.reshape(onnx_weights.shape)
        else:
            blended = onnx_weights + weights

    logger.trace("blended weight shape: %s", blended.shape)

    # replace the original initializer
    return numpy_helper.from_array(blended.astype(onnx_weights.dtype), weight_node.name)


def blend_node_matmul(matmul_node, weights, matmul_key) -> TensorProto:
    onnx_weights = numpy_helper.to_array(matmul_node)
    logger.trace(
        "found blended weights for matmul: %s, %s",
        weights.shape,
        onnx_weights.shape,
    )

    t_weights = weights.transpose()
    if weights.shape != onnx_weights.shape and t_weights.shape != onnx_weights.shape:
        logger.warning(
            "weight shapes do not match for %s: %s vs %s",
            matmul_key,
            weights.shape,
            onnx_weights.shape,
        )
        t_weights = interp_to_match(weights, onnx_weights).transpose()

    blended = onnx_weights + t_weights
    logger.trace("blended weight shape: %s, %s", blended.shape, onnx_weights.dtype)

    # replace the original initializer
    return numpy_helper.from_array(blended.astype(onnx_weights.dtype), matmul_node.name)


def blend_loras(
    _conversion: ServerContext,
    base_name: Union[str, ModelProto],
    loras: List[Tuple[str, float]],
    model_type: Literal["text_encoder", "unet"],
    model_index: Optional[int] = None,
    xl: Optional[bool] = False,
):
    # always load to CPU for blending
    device = torch.device("cpu")
    dtype = torch.float32

    base_model = base_name if isinstance(base_name, ModelProto) else load(base_name)
    lora_models = [load_tensor(name, map_location=device) for name, _weight in loras]

    if model_type == "text_encoder":
        if model_index is None:
            lora_prefix = "lora_te_"
        else:
            lora_prefix = f"lora_te{model_index}_"
    else:
        lora_prefix = f"lora_{model_type}_"

    layers = []
    for (lora_name, lora_weight), lora_model in zip(loras, lora_models):
        logger.debug("blending LoRA from %s with weight of %s", lora_name, lora_weight)
        if lora_model is None:
            logger.warning("unable to load tensor for LoRA")
            continue

        blended: Dict[str, np.ndarray] = {}
        layers.append(blended)

        for key in lora_model.keys():
            if ".hada_w1_a" in key and lora_prefix in key:
                # LoHA
                base_key, np_weights = blend_weights_loha(
                    key, lora_prefix, lora_model, dtype
                )
                np_weights = np_weights * lora_weight
                logger.trace(
                    "adding LoHA weights: %s",
                    np_weights.shape,
                )
                blended[base_key] = np_weights
            elif ".lora_down" in key and lora_prefix in key:
                # LoRA or LoCON
                base_key, np_weights = blend_weights_lora(
                    key, lora_prefix, lora_model, dtype
                )
                np_weights = np_weights * lora_weight
                logger.trace(
                    "adding LoRA weights: %s",
                    np_weights.shape,
                )
                blended[base_key] = np_weights

    # rewrite node names for XL and flatten layers
    weights = Dict[str, np.ndarray] = {}

    for blended in layers:
        if xl:
            nodes = list(base_model.graph.node)
            blended = fix_xl_names(blended, nodes)

        for key, value in blended.items():
            if key in weights:
                weights[key] = sum_weights(weights[key], value)
            else:
                weights[key] = value

    # fix node names once
    fixed_initializer_names = [
        fix_initializer_name(node.name) for node in base_model.graph.initializer
    ]
    fixed_node_names = [fix_node_name(node.name) for node in base_model.graph.node]

    logger.debug(
        "updating %s of %s initializers",
        len(weights.keys()),
        len(base_model.graph.initializer),
    )

    unmatched_keys = []
    for base_key, weights in weights.items():
        conv_key = base_key + "_Conv"
        gemm_key = base_key + "_Gemm"
        matmul_key = base_key + "_MatMul"

        logger.trace(
            "key %s has conv: %s, gemm: %s, matmul: %s",
            base_key,
            conv_key in fixed_node_names,
            gemm_key in fixed_node_names,
            matmul_key in fixed_node_names,
        )

        if conv_key in fixed_node_names or gemm_key in fixed_node_names:
            if conv_key in fixed_node_names:
                conv_idx = fixed_node_names.index(conv_key)
                conv_node = base_model.graph.node[conv_idx]
                logger.trace(
                    "found conv node %s using %s", conv_node.name, conv_node.input
                )
            else:
                conv_idx = fixed_node_names.index(gemm_key)
                conv_node = base_model.graph.node[conv_idx]
                logger.trace(
                    "found gemm node %s using %s", conv_node.name, conv_node.input
                )

            # find weight initializer
            weight_name = [n for n in conv_node.input if ".weight" in n][0]
            weight_name = fix_initializer_name(weight_name)

            weight_idx = fixed_initializer_names.index(weight_name)
            weight_node = base_model.graph.initializer[weight_idx]
            logger.trace("found weight initializer: %s", weight_node.name)

            # replace the previous node
            updated_node = blend_node_conv_gemm(weight_node, weights)

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

            # replace the previous node
            updated_node = blend_node_matmul(matmul_node, weights, matmul_key)

            del base_model.graph.initializer[matmul_idx]
            base_model.graph.initializer.insert(matmul_idx, updated_node)
        else:
            unmatched_keys.append(base_key)

    logger.trace(
        "node counts: %s -> %s, %s -> %s",
        len(fixed_initializer_names),
        len(base_model.graph.initializer),
        len(fixed_node_names),
        len(base_model.graph.node),
    )

    if len(unmatched_keys) > 0:
        logger.warning("could not find nodes for some LoRA keys: %s", unmatched_keys)

    return base_model


def interp_to_match(ref: np.ndarray, resize: np.ndarray) -> np.ndarray:
    res_x = np.linspace(0, 1, resize.shape[0])
    res_y = np.linspace(0, 1, resize.shape[1])
    ref_x = np.linspace(0, 1, ref.shape[0])
    ref_y = np.linspace(0, 1, ref.shape[1])
    logger.debug(
        "dims: %s, %s, %s, %s",
        resize.shape[0],
        resize.shape[1],
        ref.shape[0],
        ref.shape[1],
    )

    f = interpolate.RegularGridInterpolator((ref_x, ref_y), ref, method="linear")
    xg, yg = np.meshgrid(res_x, res_y)
    output = f((xg, yg))
    logger.debug("weights after interpolation: %s", output.shape)

    return output
