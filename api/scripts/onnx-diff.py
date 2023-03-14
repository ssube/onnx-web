from logging import getLogger, basicConfig, DEBUG
from onnx import load_model, ModelProto
from onnx.numpy_helper import to_array
from sys import argv, stdout


basicConfig(stream=stdout, level=DEBUG)

logger = getLogger(__name__)

def diff_models(ref_model: ModelProto, cmp_model: ModelProto):
  if len(ref_model.graph.initializer) != len(cmp_model.graph.initializer):
    logger.warning("different number of initializers: %s vs %s", len(ref_model.graph.initializer), len(cmp_model.graph.initializer))
  else:
    for (ref_init, cmp_init) in zip(ref_model.graph.initializer, cmp_model.graph.initializer):
      if ref_init.name != cmp_init.name:
        logger.info("different node names: %s vs %s", ref_init.name, cmp_init.name)
      elif ref_init.data_location != cmp_init.data_location:
        logger.info("different data locations: %s vs %s", ref_init.data_location, cmp_init.data_location)
      elif ref_init.data_type != cmp_init.data_type:
        logger.info("different data types: %s vs %s", ref_init.data_type, cmp_init.data_type)
      elif len(ref_init.raw_data) != len(cmp_init.raw_data):
        logger.info("different raw data size: %s vs %s", len(ref_init.raw_data), len(cmp_init.raw_data))
      elif len(ref_init.raw_data) > 0 and len(cmp_init.raw_data) > 0:
        ref_data = to_array(ref_init)
        cmp_data = to_array(cmp_init)
        data_diff = ref_data - cmp_data
        if data_diff.max() > 0:
          logger.info("raw data differs: %s", data_diff)
      else:
        logger.info("initializers are identical in all checked fields: %s", ref_init.name)


if __name__ == "__main__":
  ref_path = argv[1]
  cmp_paths = argv[2:]

  logger.info("loading reference model from %s", ref_path)
  ref_model = load_model(ref_path)

  for cmp_path in cmp_paths:
    logger.info("loading comparison model from %s", cmp_path)
    cmp_model = load_model(cmp_path)
    diff_models(ref_model, cmp_model)
