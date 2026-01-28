from enum import Enum


class PublicLLM(str, Enum):
    LARGE = "large"
    SMALL = "small"

    def __str__(self) -> str:
        return str(self.value)
