from logging import getLogger
from random import randint
from typing import Optional

from transformers import pipeline

from ..params import ImageParams, SizeChart, StageParams
from ..server import ServerContext
from ..worker import ProgressCallback, WorkerContext
from .base import BaseStage
from .result import StageResult

logger = getLogger(__name__)


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
        **kwargs,
    ) -> StageResult:
        gpt2_pipe = pipeline("text-generation", model=prompt_model, tokenizer="gpt2")
        gpt2_pipe = gpt2_pipe.to("cuda")

        input = params.prompt
        max_length = len(input) + randint(60, 90)
        logger.debug(
            "generating new prompt with max length of %d from input prompt: %s",
            max_length,
            input,
        )

        result = gpt2_pipe(input, max_length=max_length, num_return_sequences=1)
        prompt = result[0]["generated_text"].strip()
        logger.debug("replacing prompt with: %s", prompt)

        params.prompt = prompt
        return sources
