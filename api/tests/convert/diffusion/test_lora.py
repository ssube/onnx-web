import unittest

import numpy as np
from onnx import GraphProto, ModelProto, NodeProto
from onnx.numpy_helper import from_array

from onnx_web.convert.diffusion.lora import (
    blend_loras,
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

        self.assertEqual(len(slim_model.graph.initializer), len(model.graph.initializer))
        self.assertEqual(len(external_weights), 1)


class FixInitializerKeyTests(unittest.TestCase):
    def test_fix_name(self):
        inputs = ["lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0.lora_down.weight"]
        outputs = ["lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_out_0_lora_down_weight"]

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
        fixed = fix_xl_names(nodes, [
            NodeProto(name="/down_blocks_proj/MatMul"),
        ])

        self.assertEqual(fixed, {
            "down_blocks_proj": nodes["input_block_proj.lora_down.weight"],
        })

    def test_middle_block(self):
        nodes = {
            "middle_block_proj.lora_down.weight": {},
        }
        fixed = fix_xl_names(nodes, [
            NodeProto(name="/mid_blocks_proj/MatMul"),
        ])

        self.assertEqual(fixed, {
            "mid_blocks_proj": nodes["middle_block_proj.lora_down.weight"],
        })

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
            "output_block_proj_o.lora_down.weight": {},
        }
        fixed = fix_xl_names(nodes, [
            NodeProto(name="/up_blocks_proj_o/MatMul"),
        ])

        self.assertEqual(fixed, {
            "up_blocks_proj_out": nodes["output_block_proj_o.lora_down.weight"],
        })


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

class BlendLoRATests(unittest.TestCase):
    pass

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
