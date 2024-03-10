from typing import List, Optional


class PromptNetwork:
    type: str
    name: str
    strength: float

    def __init__(self, type: str, name: str, strength: float) -> None:
        self.type = type
        self.name = name
        self.strength = strength

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.type == self.type
            and other.name == self.name
            and other.strength == self.strength
        )

    def __repr__(self) -> str:
        return f"PromptNetwork({self.type}, {self.name}, {self.strength})"


class PromptPhrase:
    phrase: str
    weight: float

    def __init__(self, phrase: str, weight: float = 1.0) -> None:
        self.phrase = phrase
        self.weight = weight

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.phrase == self.phrase
            and other.weight == self.weight
        )

    def __repr__(self) -> str:
        return f"PromptPhrase({self.phrase}, {self.weight})"


class PromptRegion:
    top: int
    left: int
    bottom: int
    right: int
    prompt: str
    append: bool

    def __init__(
        self,
        top: int,
        left: int,
        bottom: int,
        right: int,
        prompt: str,
        append: bool,
    ) -> None:
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.prompt = prompt
        self.append = append

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.top == self.top
            and other.left == self.left
            and other.bottom == self.bottom
            and other.right == self.right
            and other.prompt == self.prompt
            and other.append == self.append
        )

    def __repr__(self) -> str:
        return f"PromptRegion({self.top}, {self.left}, {self.bottom}, {self.right}, {self.prompt}, {self.append})"


class PromptSeed:
    top: int
    left: int
    bottom: int
    right: int
    seed: int

    def __init__(self, top: int, left: int, bottom: int, right: int, seed: int) -> None:
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.seed = seed

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.top == self.top
            and other.left == self.left
            and other.bottom == self.bottom
            and other.right == self.right
            and other.seed == self.seed
        )

    def __repr__(self) -> str:
        return f"PromptSeed({self.top}, {self.left}, {self.bottom}, {self.right}, {self.seed})"


class Prompt:
    clip_skip: int
    networks: List[PromptNetwork]
    positive_phrases: List[PromptPhrase]
    negative_phrases: List[PromptPhrase]
    region_prompts: List[PromptRegion]
    region_seeds: List[PromptSeed]

    def __init__(
        self,
        networks: Optional[List[PromptNetwork]],
        positive_phrases: List[PromptPhrase],
        negative_phrases: List[PromptPhrase],
        region_prompts: List[PromptRegion],
        region_seeds: List[PromptSeed],
        clip_skip: int,
    ) -> None:
        self.positive_phrases = positive_phrases
        self.negative_prompt = negative_phrases
        self.networks = networks or []
        self.region_prompts = region_prompts or []
        self.region_seeds = region_seeds or []
        self.clip_skip = clip_skip

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, self.__class__)
            and other.networks == self.networks
            and other.positive_phrases == self.positive_phrases
            and other.negative_phrases == self.negative_phrases
            and other.region_prompts == self.region_prompts
            and other.region_seeds == self.region_seeds
            and other.clip_skip == self.clip_skip
        )

    def __repr__(self) -> str:
        return f"Prompt({self.networks}, {self.positive_phrases}, {self.negative_phrases}, {self.region_prompts}, {self.region_seeds}, {self.clip_skip})"
