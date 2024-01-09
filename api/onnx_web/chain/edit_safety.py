from logging import getLogger
from typing import Optional

from PIL import Image

from ..errors import CancelledException
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


class EditSafetyStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        _worker: WorkerContext,
        server: ServerContext,
        _stage: StageParams,
        _params: ImageParams,
        sources: StageResult,
        *,
        callback: Optional[ProgressCallback] = None,
        **kwargs,
    ) -> StageResult:
        logger.info("checking results using horde safety")

        # keep these within run to make this sort of like a plugin or peer dependency
        try:
            from horde_safety.deep_danbooru_model import get_deep_danbooru_model
            from horde_safety.interrogate import get_interrogator_no_blip
            from horde_safety.nsfw_checker_class import NSFWChecker

            # set up
            block_nsfw = server.has_feature("horde-safety-nsfw")

            interrogator = get_interrogator_no_blip()
            deep_danbooru_model = get_deep_danbooru_model()

            nsfw_checker = NSFWChecker(
                interrogator,
                deep_danbooru_model,
            )

            # individual flags from NSFWResult
            is_csam = False

            images = sources.as_images()
            results = []
            for i, image in enumerate(images):
                prompt = sources.metadata[i].params.prompt
                check = nsfw_checker.check_for_nsfw(image, prompt=prompt)

                if check.is_csam:
                    logger.warning("flagging csam result: %s, %s", i, prompt)
                    is_csam = True

                if check.is_nsfw and block_nsfw:
                    logger.warning("blocking nsfw image: %s, %s", i, prompt)
                    results.append(Image.new("RGB", image.size, color="black"))

            if is_csam:
                logger.warning("blocking csam result")
                raise CancelledException(reason="csam")
            else:
                return StageResult.from_images(results, metadata=sources.metadata)
        except ImportError:
            logger.warning("horde safety not installed")
            return StageResult.empty()
