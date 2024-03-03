"""
Patches for optimum's internal conversion process.
"""

from logging import getLogger

from optimum.exporters.onnx import model_patcher

logger = getLogger(__name__)

original_override_arguments = model_patcher.override_arguments


def override_override_arguments(args, kwargs, signature, model_kwargs=None):
    """
    Override the arguments of the `override_arguments` function.
    """
    logger.debug(
        "overriding arguments for `override_arguments`: %s, %s, %s",
        args,
        kwargs,
        signature,
    )

    if "output_hidden_states" in signature.parameters:
        logger.debug("enabling hidden states for model")
        parameter_names = list(signature.parameters.keys())
        hidden_states_index = parameter_names.index("output_hidden_states")

        # convert the arguments to a list for modification
        arg_list = list(args)
        arg_list[hidden_states_index] = True
        args = tuple(arg_list)

    return original_override_arguments(args, kwargs, signature, model_kwargs)


def patch_optimum():
    logger.info("installing patches for optimum's internal conversion process")
    model_patcher.override_arguments = override_override_arguments
