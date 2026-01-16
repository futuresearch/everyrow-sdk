from enum import Enum


class TaskType(str, Enum):
    AGENT = "agent"
    CONCATENATE = "concatenate"
    CREATE = "create"
    CREATE_GROUP = "create_group"
    DEDUPE = "dedupe"
    DEEP_MERGE = "deep_merge"
    DEEP_RANK = "deep_rank"
    DEEP_SCREEN = "deep_screen"
    DERIVE = "derive"
    DROP_COLUMNS = "drop_columns"
    FILTER = "filter"
    FIND_NUMBER = "find_number"
    FIND_SOURCES = "find_sources"
    FLATTEN = "flatten"
    GET_NUMBER = "get_number"
    GROUP_BY = "group_by"
    IMPORT_SHEET = "import_sheet"
    JOIN = "join"
    MULTIAGENT = "multiagent"
    POPULATE_REFERENCE_CLASS = "populate_reference_class"
    REVISE = "revise"
    UPLOAD_CSV = "upload_csv"
    VALIDATE_CLAIM = "validate_claim"
    VALUE_23 = "_provide_workflow_inputs"

    def __str__(self) -> str:
        return str(self.value)
