from importlib import import_module
from logging import getLogger
from typing import Any, Callable, Dict

from ..chain.stages import add_stage
from ..diffusers.load import add_pipeline
from ..server.context import ServerContext

logger = getLogger(__name__)


class PluginExports:
    clients: Dict[str, Any]
    converter: Dict[str, Any]
    pipelines: Dict[str, Any]
    stages: Dict[str, Any]

    def __init__(
        self, clients=None, converter=None, pipelines=None, stages=None
    ) -> None:
        self.clients = clients or {}
        self.converter = converter or {}
        self.pipelines = pipelines or {}
        self.stages = stages or {}


PluginModule = Callable[[ServerContext], PluginExports]


def load_plugins(server: ServerContext) -> PluginExports:
    combined_exports = PluginExports()

    for plugin in server.plugins:
        logger.info("loading plugin module: %s", plugin)
        try:
            module: PluginModule = import_module(plugin)
            exports = module(server)

            for name, pipeline in exports.pipelines.items():
                if name in combined_exports.pipelines:
                    logger.warning(
                        "multiple plugins exported a pipeline named %s", name
                    )
                else:
                    combined_exports.pipelines[name] = pipeline

            for name, stage in exports.stages.items():
                if name in combined_exports.stages:
                    logger.warning("multiple plugins exported a stage named %s", name)
                else:
                    combined_exports.stages[name] = stage
        except Exception:
            logger.exception("error importing plugin")

    return combined_exports


def register_plugins(exports: PluginExports) -> bool:
    success = True

    for name, pipeline in exports.pipelines.items():
        success = success and add_pipeline(name, pipeline)

    for name, stage in exports.stages.items():
        success = success and add_stage(name, stage)

    return success
