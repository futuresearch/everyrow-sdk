from enum import Enum


class EventType(str, Enum):
    ARTIFACTCHANGED = "artifact:changed"
    ARTIFACT_PROGRESSCHANGED = "artifact_progress:changed"
    CONVERSATIONCHANGED = "conversation:changed"
    MESSAGECREATED = "message:created"
    SESSIONCHANGED = "session:changed"
    TASKCHANGED = "task:changed"
    TRACECHANGED = "trace:changed"

    def __str__(self) -> str:
        return str(self.value)
