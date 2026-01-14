from enum import Enum


class ArtifactStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    OMITTED_FROM_PREVIEW = "omitted_from_preview"
    PENDING = "pending"
    REVOKED = "revoked"
    RUNNING = "running"
    UNDER_REVIEW = "under_review"

    def __str__(self) -> str:
        return str(self.value)
