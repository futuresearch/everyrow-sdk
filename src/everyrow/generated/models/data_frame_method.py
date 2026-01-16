from enum import Enum


class DataFrameMethod(str, Enum):
    EQ = "eq"
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    NE = "ne"
    NLARGEST = "nlargest"
    NSMALLEST = "nsmallest"
    STR_CONTAINS = "str.contains"
    STR_ENDSWITH = "str.endswith"
    STR_STARTSWITH = "str.startswith"

    def __str__(self) -> str:
        return str(self.value)
