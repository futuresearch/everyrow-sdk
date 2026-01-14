from enum import Enum


class ResponseSchemaType(str, Enum):
    CUSTOM = "CUSTOM"
    JSON = "JSON"

    def __str__(self) -> str:
        return str(self.value)
