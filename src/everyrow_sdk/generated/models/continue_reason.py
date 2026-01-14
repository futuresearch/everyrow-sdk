from enum import Enum


class ContinueReason(str, Enum):
    FAILURE = "failure"
    FROM_PREVIEW = "from_preview"

    def __str__(self) -> str:
        return str(self.value)
