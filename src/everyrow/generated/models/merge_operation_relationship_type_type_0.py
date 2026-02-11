from enum import Enum


class MergeOperationRelationshipTypeType0(str, Enum):
    MANY_TO_ONE = "many_to_one"
    ONE_TO_ONE = "one_to_one"

    def __str__(self) -> str:
        return str(self.value)
