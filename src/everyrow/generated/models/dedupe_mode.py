from enum import Enum


class DedupeMode(str, Enum):
    AGENTIC = "agentic"
    DIRECT = "direct"

    def __str__(self) -> str:
        return str(self.value)
