from enum import Enum


class SubmitChatTaskBodySelectedTaskTypeType0(str, Enum):
    DEDUPE = "dedupe"
    DEEP_MERGE = "deep_merge"
    DEEP_RANK = "deep_rank"
    DEEP_SCREEN = "deep_screen"

    def __str__(self) -> str:
        return str(self.value)
