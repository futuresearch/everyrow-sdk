from enum import Enum


class AgentTaskArgsProcessingMode(str, Enum):
    MAP = "map"
    REDUCE = "reduce"

    def __str__(self) -> str:
        return str(self.value)
