from typing import List, Optional


class NetworkWeight:
    pass


class PromptRegion:
    pass


class PromptSeed:
    pass


class StructuredPrompt:
    prompt: str
    negative_prompt: Optional[str]
    networks: List[NetworkWeight]
    region_prompts: List[PromptRegion]
    region_seeds: List[PromptSeed]

    def __init__(
        self, prompt: str, negative_prompt: Optional[str], networks: List[NetworkWeight]
    ) -> None:
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.networks = networks or []
        self.region_prompts = []
        self.region_seeds = []
