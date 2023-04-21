from logging import getLogger, basicConfig, DEBUG
from onnx import load_model, ModelProto
from onnx.numpy_helper import to_array
from sys import argv, stdout


basicConfig(stream=stdout, level=DEBUG)

logger = getLogger(__name__)

def diff_models(ref_model: ModelProto, cmp_model: ModelProto):
  diffs = 0

  if len(ref_model.graph.initializer) != len(cmp_model.graph.initializer):
    logger.warning("different number of initializers: %s vs %s", len(ref_model.graph.initializer), len(cmp_model.graph.initializer))
    diffs += abs(len(ref_model.graph.initializer) - len(cmp_model.graph.initializer))
  else:
    for (ref_init, cmp_init) in zip(ref_model.graph.initializer, cmp_model.graph.initializer):
      if ref_init.name != cmp_init.name:
        logger.info("different node names: %s vs %s", ref_init.name, cmp_init.name)
        diffs += 1
      elif ref_init.data_location != cmp_init.data_location:
        logger.info("different data locations: %s vs %s", ref_init.data_location, cmp_init.data_location)
        diffs += 1
      elif ref_init.data_type != cmp_init.data_type:
        logger.info("different data types: %s vs %s", ref_init.data_type, cmp_init.data_type)
        diffs += 1
      elif len(ref_init.raw_data) != len(cmp_init.raw_data):
        ref_data = to_array(ref_init)
        cmp_data = to_array(cmp_init)
        logger.info("different raw data shapes: %s vs %s", ref_data.shape, cmp_data.shape)
        diffs += 1
      elif len(ref_init.raw_data) > 0 and len(cmp_init.raw_data) > 0:
        ref_data = to_array(ref_init)
        cmp_data = to_array(cmp_init)
        data_diff = ref_data - cmp_data
        if data_diff.max() != 0:
          logger.info("raw data differs for %s: %s", ref_init.name, data_diff.max())
          diffs += 1
      else:
        logger.info("initializers are identical in all checked fields: %s", ref_init.name)

  if len(ref_model.graph.node) != len(cmp_model.graph.node):
    logger.warning("different number of nodes: %s vs %s", len(ref_model.graph.node), len(cmp_model.graph.node))
    diffs += abs(len(ref_model.graph.node) - len(cmp_model.graph.node))
  else:
    for (ref_node, cmp_node) in zip(ref_model.graph.node, cmp_model.graph.node):
      if ref_node.name != cmp_node.name:
        logger.info("different node names: %s vs %s", ref_node.name, cmp_node.name)
        diffs += 1
      elif ref_node.input != cmp_node.input:
        logger.info("different inputs: %s vs %s", ref_node.input, cmp_node.input)
        diffs += 1
      elif ref_node.output != cmp_node.output:
        logger.info("different outputs: %s vs %s", ref_node.output, cmp_node.output)
        diffs += 1
      elif ref_node.op_type != cmp_node.op_type:
        logger.info("different op type: %s vs %s", ref_node.op_type, cmp_node.op_type)
        diffs += 1
      else:
        logger.info("nodes are identical in all checked fields: %s", ref_node.name)

  if diffs > 0:
    logger.warning("models have %s differences", diffs)
  else:
    logger.info("models have no detectable differences")


if __name__ == "__main__":
  ref_path = argv[1]
  cmp_paths = argv[2:]

  logger.info("loading reference model from %s", ref_path)
  ref_model = load_model(ref_path)

  for cmp_path in cmp_paths:
    logger.info("loading comparison model from %s", cmp_path)
    cmp_model = load_model(cmp_path)
    diff_models(ref_model, cmp_model)
