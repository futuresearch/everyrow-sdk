from enum import Enum


class AgentQueryParamsSystemPromptKindType0(str, Enum):
    PERSISTENT = "persistent"
    SPEED_FOCUSED = "speed_focused"
    TONED_DOWN_WITH_ALLOWED_CONFLICTS = "toned_down_with_allowed_conflicts"

    def __str__(self) -> str:
        return str(self.value)
