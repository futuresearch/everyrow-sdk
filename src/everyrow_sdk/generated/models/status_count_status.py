from enum import Enum


class StatusCountStatus(str, Enum):
    FAILURE = "FAILURE"
    PENDING = "PENDING"
    RETRY = "RETRY"
    REVOKED = "REVOKED"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"

    def __str__(self) -> str:
        return str(self.value)
