from typing import List, Literal

NetworkType = Literal["control", "inversion", "lora"]


class NetworkModel:
    name: str
    tokens: List[str]
    type: NetworkType

    def __init__(self, name: str, type: NetworkType, tokens=None) -> None:
        self.name = name
        self.tokens = tokens or []
        self.type = type

    def tojson(self):
        return {
            "name": self.name,
            "tokens": self.tokens,
            "type": self.type,
        }
