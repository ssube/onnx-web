from logging import getLogger
from typing import Any, Optional

from PIL import Image

from ..errors import CancelledException
from ..output import save_metadata
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..server.model_cache import ModelTypes
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class EditSafetyStage(BaseStage):
    max_tile = SizeChart.max

    def load(self, server: ServerContext, device: str) -> Any:
        # keep these within run to make this sort of like a plugin or peer dependency
        from horde_safety.deep_danbooru_model import get_deep_danbooru_model
        from horde_safety.interrogate import get_interrogator_no_blip
        from horde_safety.nsfw_checker_class import NSFWChecker

        # check cache
        cache_key = ("horde-safety",)
        cache_checker = server.cache.get(ModelTypes.safety, cache_key)
        if cache_checker is not None:
            return cache_checker

        # set up
        interrogator = get_interrogator_no_blip(device=device)
        deep_danbooru_model = get_deep_danbooru_model(device=device)

        nsfw_checker = NSFWChecker(
            interrogator,
            deep_danbooru_model,
        )

        server.cache.set(ModelTypes.safety, cache_key, nsfw_checker)

        return nsfw_checker

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("checking results using horde safety")

        try:
            # set up
            torch_device = worker.device.torch_str()
            nsfw_checker = self.load(server, torch_device)
            block_nsfw = server.has_feature("horde-safety-nsfw")
            is_csam = False

            # check each output
            images = sources.as_images()
            results = []
            for i, image in enumerate(images):
                metadata = sources.metadata[i]
                prompt = metadata.params.prompt
                check = nsfw_checker.check_for_nsfw(image, prompt=prompt)

                if check.is_csam:
                    logger.warning("flagging csam result: %s, %s", i, prompt)
                    is_csam = True

                    report_name = f"csam-report-{worker.job}-{i}"
                    report_path = save_metadata(server, report_name, metadata)
                    logger.info("saved csam report: %s", report_path)
                elif check.is_nsfw and block_nsfw:
                    logger.warning("blocking nsfw image: %s, %s", i, prompt)
                    results.append(Image.new("RGB", image.size, color="black"))
                else:
                    results.append(image)

            if is_csam:
                logger.warning("blocking csam result")
                raise CancelledException(reason="csam")
            else:
                return StageResult.from_images(results, metadata=sources.metadata)
        except ImportError:
            logger.warning("horde safety not installed")
            return StageResult.empty()
