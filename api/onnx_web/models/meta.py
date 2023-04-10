from typing import Literal

NetworkType = Literal["inversion", "lora"]


class NetworkModel:
    name: str
    type: NetworkType

    def __init__(self, name: str, type: NetworkType) -> None:
        self.name = name
        self.type = type

    def tojson(self):
        return {
            "name": self.name,
            "type": self.type,
        }
