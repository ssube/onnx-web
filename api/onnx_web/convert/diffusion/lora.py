from numpy import ndarray
from onnx import TensorProto, helper, load, numpy_helper, ModelProto, save_model
from typing import Dict, List, Tuple
from logging import getLogger


logger = getLogger(__name__)


def load_lora(filename: str):
  model = load(filename)

  for weight in model.graph.initializer:
    # print(weight.name, numpy_helper.to_array(weight).shape)
    pass

  return model


def blend_loras(base: ModelProto, weights: List[ModelProto], alphas: List[float]) -> List[Tuple[TensorProto, ndarray]]:
  total = 1 + sum(alphas)

  results = []

  for base_node in base.graph.initializer:
    logger.info("blending initializer node %s", base_node.name)
    base_weights = numpy_helper.to_array(base_node).copy()

    for weight, alpha in zip(weights, alphas):
      weight_node = next(iter([f for f in weight.graph.initializer if f.name == base_node.name]), None)

      if weight_node is not None:
        base_weights += numpy_helper.to_array(weight_node) * alpha
      else:
        logger.warning("missing weights: %s in %s", base_node.name, weight.doc_string)

    results.append((base_node, base_weights / total))

  return results


def convert_diffusion_lora(part: str):
  lora_weights = [
    f"diffusion-lora-jack/{part}/model.onnx",
    f"diffusion-lora-taters/{part}/model.onnx",
  ]

  base = load_lora(f"stable-diffusion-onnx-v1-5/{part}/model.onnx")
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

  save_model(
    model,
    f"/tmp/lora-{part}.onnx",
    save_as_external_data=True,
    all_tensors_to_one_file=True,
    location=f"/tmp/lora-{part}.tensors",
  )
  logger.info("saved model to %s and tensors to %s", f"/tmp/lora-{part}.onnx", f"/tmp/lora-{part}.tensors")


if __name__ == "__main__":
  convert_diffusion_lora("unet")
  convert_diffusion_lora("text_encoder")