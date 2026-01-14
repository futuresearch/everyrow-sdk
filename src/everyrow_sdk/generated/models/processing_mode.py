from enum import Enum


class ProcessingMode(str, Enum):
    MAP = "map"
    REDUCE = "reduce"
    TRANSFORM = "transform"

    def __str__(self) -> str:
        return str(self.value)
