from enum import Enum


class PublicTaskType(str, Enum):
    AGENT = "agent"
    DEDUPE = "dedupe"
    MERGE = "merge"
    RANK = "rank"
    SCREEN = "screen"

    def __str__(self) -> str:
        return str(self.value)
