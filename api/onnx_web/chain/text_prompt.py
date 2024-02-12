from logging import getLogger
from random import randint
from re import sub
from typing import Optional

from transformers import pipeline

from ..diffusers.utils import split_prompt
from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


LENGTH_MARGIN = 15
RETRY_LIMIT = 5


class TextPromptStage(BaseStage):
    max_tile = SizeChart.max

    def run(
        self,
        worker: WorkerContext,
        server: ServerContext,
        stage: StageParams,
        params: ImageParams,
        sources: StageResult,
        *,
        callback: Optional[ProgressCallback] = None,
        prompt_model: str = "Gustavosta/MagicPrompt-Stable-Diffusion",
        exclude_tokens: Optional[str] = None,
        min_length: int = 75,
        **kwargs,
    ) -> StageResult:
        device = worker.device.torch_str()
        text_pipe = pipeline(
            "text-generation",
            model=prompt_model,
            device=device,
            framework="pt",
        )

        prompt_parts = split_prompt(params.prompt)
        prompt_results = []
        for prompt in prompt_parts:
            retries = 0
            while len(prompt) < min_length and retries < RETRY_LIMIT:
                max_length = len(prompt) + randint(
                    min_length - LENGTH_MARGIN, min_length + LENGTH_MARGIN
                )
                logger.debug(
                    "extending input prompt to max length of %d from %s: %s",
                    max_length,
                    len(prompt),
                    prompt,
                )

                result = text_pipe(
                    prompt, max_length=max_length, num_return_sequences=1
                )
                prompt = result[0]["generated_text"].strip()

                if exclude_tokens:
                    logger.debug(
                        "removing excluded tokens from prompt: %s", exclude_tokens
                    )
                    prompt = sub(exclude_tokens, "", prompt)

            if retries >= RETRY_LIMIT:
                logger.warning(
                    "failed to extend input prompt to min length of %d, ended up with %d: %s",
                    min_length,
                    len(prompt),
                    prompt,
                )

            prompt_results.append(prompt)

        complete_prompt = " || ".join(prompt_results)
        logger.debug("replacing input prompt: %s -> %s", params.prompt, complete_prompt)
        params.prompt = complete_prompt
        return sources
