from enum import Enum


class TaskEffort(str, Enum):
    HIGH = "high"
    LOW = "low"
    MINIMAL = "minimal"

    def __str__(self) -> str:
        return str(self.value)
