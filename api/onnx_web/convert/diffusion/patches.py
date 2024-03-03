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
    logger.info(
        "overriding arguments for `override_arguments`: %s, %s, %s",
        args,
        kwargs,
        signature,
    )

    # if "return_hidden_states" signature.parameters:
    #     args[4] = True

    return original_override_arguments(args, kwargs, signature, model_kwargs)


def patch_optimum():
    logger.info("installing patches for optimum's internal conversion process")
    model_patcher.override_arguments = override_override_arguments
