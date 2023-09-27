from argparse import ArgumentParser
from onnx_web.convert.diffusion.lora import blend_loras, buffer_external_data_tensors
from os import path
from onnx.checker import check_model
from onnx.external_data_helper import (
    convert_model_to_external_data,
    write_external_data_tensors,
)
from onnxruntime import InferenceSession, SessionOptions
from logging import getLogger

from onnx_web.convert.utils import ConversionContext

logger = getLogger(__name__)


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
