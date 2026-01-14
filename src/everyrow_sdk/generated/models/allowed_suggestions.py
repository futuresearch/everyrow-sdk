from enum import Enum


class AllowedSuggestions(str, Enum):
    RUN_ON_ALL_ROWS = "run_on_all_rows"

    def __str__(self) -> str:
        return str(self.value)
