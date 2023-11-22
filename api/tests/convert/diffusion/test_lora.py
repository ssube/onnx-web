import unittest

import numpy as np
import torch
from onnx import GraphProto, ModelProto, NodeProto
from onnx.numpy_helper import from_array

from onnx_web.convert.diffusion.lora import (
    blend_node_conv_gemm,
    blend_node_matmul,
    blend_weights_loha,
    blend_weights_lora,
    buffer_external_data_tensors,
    fix_initializer_name,
    fix_node_name,
    fix_xl_names,
    interp_to_match,
    kernel_slice,
    sum_weights,
)


class SumWeightsTests(unittest.TestCase):
    def test_same_shape(self):
        weights = sum_weights(np.zeros((4, 4)), np.ones((4, 4)))
        self.assertEqual(weights.shape, (4, 4))

    def test_1x1_kernel(self):
        weights = sum_weights(np.zeros((4, 4, 1, 1)), np.ones((4, 4)))
        self.assertEqual(weights.shape, (4, 4, 1, 1))

        weights = sum_weights(np.zeros((4, 4)), np.ones((4, 4, 1, 1)))
        self.assertEqual(weights.shape, (4, 4, 1, 1))

    def test_3x3_kernel(self):
        """
        weights = sum_weights(np.zeros((4, 4, 3, 3)), np.ones((4, 4)))
        self.assertEqual(weights.shape, (4, 4, 1, 1))
        """
        pass


class BufferExternalDataTensorTests(unittest.TestCase):
    def test_basic_external(self):
        model = ModelProto(
            graph=GraphProto(
                initializer=[
                    from_array(np.zeros((4, 4))),
                ],
            )
        )
        (slim_model, external_weights) = buffer_external_data_tensors(model)

        self.assertEqual(
            len(slim_model.graph.initializer), len(model.graph.initializer)
        )
        self.assertEqual(len(external_weights), 1)


class FixInitializerKeyTests(unittest.TestCase):
    def test_fix_name(self):
        inputs = [
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight"
        ]
        outputs = [
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0_lora_down_weight"
        ]

        for input, output in zip(inputs, outputs):
            self.assertEqual(fix_initializer_name(input), output)


class FixNodeNameTests(unittest.TestCase):
    def test_fix_name(self):
        inputs = [
            "lora_unet/up_blocks/3/attentions/2/transformer_blocks/0/attn2_to_out/0.lora_down.weight",
            "_prefix",
        ]
        outputs = [
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0_lora_down_weight",
            "prefix",
        ]

        for input, output in zip(inputs, outputs):
            self.assertEqual(fix_node_name(input), output)


class FixXLNameTests(unittest.TestCase):
    def test_empty(self):
        nodes = {}
        fixed = fix_xl_names(nodes, [])

        self.assertEqual(fixed, {})

    def test_input_block(self):
        nodes = {
            "input_block_proj.lora_down.weight": {},
        }
        fixed = fix_xl_names(
            nodes,
            [
                NodeProto(name="/down_blocks_proj/MatMul"),
            ],
        )

        self.assertEqual(
            fixed,
            {
                "down_blocks_proj": nodes["input_block_proj.lora_down.weight"],
            },
        )

    def test_middle_block(self):
        nodes = {
            "middle_block_proj.lora_down.weight": {},
        }
        fixed = fix_xl_names(
            nodes,
            [
                NodeProto(name="/mid_blocks_proj/MatMul"),
            ],
        )

        self.assertEqual(
            fixed,
            {
                "mid_blocks_proj": nodes["middle_block_proj.lora_down.weight"],
            },
        )

    def test_output_block(self):
        pass

    def test_text_model(self):
        pass

    def test_unknown_block(self):
        pass

    def test_unmatched_block(self):
        nodes = {
            "lora_unet.input_block.lora_down.weight": {},
        }
        fixed = fix_xl_names(nodes, [""])

        self.assertEqual(fixed, nodes)

    def test_output_projection(self):
        nodes = {
            "output_block_proj_out.lora_down.weight": {},
        }
        fixed = fix_xl_names(
            nodes,
            [
                NodeProto(name="/up_blocks_proj_out/MatMul"),
            ],
        )

        self.assertEqual(
            fixed,
            {
                "up_blocks_proj_out": nodes["output_block_proj_out.lora_down.weight"],
            },
        )


class KernelSliceTests(unittest.TestCase):
    def test_within_kernel(self):
        self.assertEqual(
            kernel_slice(1, 1, (3, 3, 3, 3)),
            (1, 1),
        )

    def test_outside_kernel(self):
        self.assertEqual(
            kernel_slice(9, 9, (3, 3, 3, 3)),
            (2, 2),
        )


class InterpToMatchTests(unittest.TestCase):
    def test_same_shape(self):
        ref = np.zeros((4, 4))
        resize = np.zeros((4, 4))
        self.assertEqual(interp_to_match(ref, resize).shape, (4, 4))

    def test_different_one_dim(self):
        ref = np.zeros((4, 2))
        resize = np.zeros((4, 4))
        self.assertEqual(interp_to_match(ref, resize).shape, (4, 4))

    def test_different_both_dims(self):
        ref = np.zeros((2, 2))
        resize = np.zeros((4, 4))
        self.assertEqual(interp_to_match(ref, resize).shape, (4, 4))


class BlendLoRATests(unittest.TestCase):
    def test_blend_unet(self):
        """
        blend_loras(None, "test", [], "unet")
        """
        pass

    def test_blend_text_encoder(self):
        """
        blend_loras(None, "test", [], "text_encoder")
        """
        pass

    def test_blend_text_encoder_index(self):
        """
        blend_loras(None, "test", [], "text_encoder", model_index=2)
        """
        pass

    def test_unmatched_keys(self):
        pass

    def test_xl_keys(self):
        """
        blend_loras(None, "test", [], "unet", xl=True)
        """
        pass

    def test_node_dtype(self):
        pass


class BlendWeightsLoHATests(unittest.TestCase):
    def test_blend_t1_t2(self):
        # blend einsum: i j k l, j r, i p -> p r k l
        i = 32
        j = 4
        k = 1
        l = 1  # NOQA
        p = 2
        r = 4

        model = {
            "foo.hada_t1": torch.from_numpy(np.ones((i, j, k, l))),
            "foo.hada_t2": torch.from_numpy(np.ones((i, j, k, l))),
            "foo.hada_w1_a": torch.from_numpy(np.ones((i, p))),
            "foo.hada_w1_b": torch.from_numpy(np.ones((j, r))),
            "foo.hada_w2_a": torch.from_numpy(np.ones((i, p))),
            "foo.hada_w2_b": torch.from_numpy(np.ones((j, r))),
            "foo.alpha": torch.tensor(1),
        }
        key, result = blend_weights_loha("foo.hada_w1_a", "", model, torch.float32)
        self.assertEqual(result.shape, (p, r, k, l))

    def test_blend_w1_w2(self):
        model = {
            "foo.hada_w1_a": torch.from_numpy(np.ones((4, 1))),
            "foo.hada_w1_b": torch.from_numpy(np.ones((1, 4))),
            "foo.hada_w2_a": torch.from_numpy(np.ones((4, 1))),
            "foo.hada_w2_b": torch.from_numpy(np.ones((1, 4))),
            "foo.alpha": torch.tensor(1),
        }
        key, result = blend_weights_loha("foo.hada_w1_a", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4))

    def test_blend_no_dim(self):
        """
        model = {
            "foo.hada_w1_a": torch.from_numpy(np.ones((1, 4))),
            "foo.hada_w1_b": torch.from_numpy(np.ones((4, 1))),
            "foo.hada_w2_a": torch.from_numpy(np.ones((1, 4))),
            "foo.hada_w2_b": torch.from_numpy(np.ones((4, 1))),
        }
        result = blend_weights_loha("foo.hada_w1_a", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4))
        """


class BlendWeightsLoRATests(unittest.TestCase):
    def test_blend_kernel_none(self):
        model = {
            "foo.lora_down": torch.from_numpy(np.ones((1, 4))),
            "foo.lora_up": torch.from_numpy(np.ones((4, 1))),
            "foo.alpha": 1,
        }
        key, result = blend_weights_lora("foo.lora_down", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4))

    def test_blend_kernel_1x1(self):
        model = {
            "foo.lora_down": torch.from_numpy(np.ones((1, 4, 1, 1))),
            "foo.lora_up": torch.from_numpy(np.ones((4, 1, 1, 1))),
            "foo.alpha": 1,
        }
        key, result = blend_weights_lora("foo.lora_down", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4, 1, 1))

    def test_blend_kernel_3x3(self):
        model = {
            "foo.lora_down": torch.from_numpy(np.ones((1, 4, 3, 3))),
            "foo.lora_up": torch.from_numpy(np.ones((4, 1, 3, 3))),
            "foo.alpha": 1,
        }
        key, result = blend_weights_lora("foo.lora_down", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4, 3, 3))

    def test_blend_kernel_3x3_cp_decomp(self):
        model = {
            "foo.lora_down": torch.from_numpy(np.ones((2, 4, 1, 1))),
            "foo.lora_mid": torch.from_numpy(np.ones((2, 2, 3, 3))),
            "foo.lora_up": torch.from_numpy(np.ones((4, 2, 1, 1))),
            "foo.alpha": 1,
        }
        key, result = blend_weights_lora("foo.lora_down", "", model, torch.float32)
        self.assertEqual(result.shape, (4, 4, 3, 3))

    def test_blend_unknown(self):
        pass


class BlendNodeConvGemmTests(unittest.TestCase):
    def test_blend_kernel_1x1_and_1x1(self):
        node = from_array(np.ones((4, 4, 1, 1)))
        result = blend_node_conv_gemm(node, np.ones((4, 4, 1, 1)))

        self.assertEqual(result.dims, [4, 4, 1, 1])
        self.assertEqual(len(result.raw_data), 4 * 4 * 8)

    def test_blend_kernel_1x1_and_none(self):
        node = from_array(np.ones((4, 4, 1, 1)))
        result = blend_node_conv_gemm(node, np.ones((4, 4)))

        self.assertEqual(result.dims, [4, 4, 1, 1])
        self.assertEqual(len(result.raw_data), 4 * 4 * 8)

    def test_blend_other_matching(self):
        node = from_array(np.ones((4, 4)))
        result = blend_node_conv_gemm(node, np.ones((4, 4)))

        self.assertEqual(result.dims, [4, 4])
        self.assertEqual(len(result.raw_data), 4 * 4 * 8)

    def test_blend_other_mismatched(self):
        pass


class BlendNodeMatMulTests(unittest.TestCase):
    def test_blend_matching(self):
        node = from_array(np.ones((4, 4)))
        result = blend_node_matmul(node, np.ones((4, 4)), "test")

        self.assertEqual(result.dims, [4, 4])
        self.assertEqual(len(result.raw_data), 4 * 4 * 8)

    def test_blend_mismatched(self):
        node = from_array(np.ones((4, 4)))
        result = blend_node_matmul(node, np.ones((2, 2)), "test")

        self.assertEqual(result.dims, [4, 4])
        self.assertEqual(len(result.raw_data), 4 * 4 * 8)
