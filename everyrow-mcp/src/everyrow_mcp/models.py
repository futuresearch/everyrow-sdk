"""Input models and schema helpers for everyrow MCP tools."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    field_validator,
    model_validator,
)

from everyrow_mcp.utils import validate_csv_output_path, validate_csv_path

PREVIEW_SIZE = 5

JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON schema dict to a dynamic Pydantic model.

    This allows the MCP client to pass arbitrary response schemas without
    needing to define Python classes.
    """
    properties = schema.get("properties", schema)
    required = set(schema.get("required", []))

    fields: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        if field_name.startswith("_") or not isinstance(field_def, dict):
            continue

        field_type_str = field_def.get("type", "string")
        python_type = JSON_TYPE_MAP.get(field_type_str, str)
        description = field_def.get("description", "")

        if field_name in required:
            fields[field_name] = (python_type, Field(..., description=description))
        else:
            fields[field_name] = (
                python_type | None,
                Field(default=None, description=description),
            )

    return create_model(name, **fields)


class AgentInput(BaseModel):
    """Input for the agent operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language task to perform on each row.", min_length=1
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the agent's response per row.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "AgentInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class SingleAgentInput(BaseModel):
    """Input for the single agent operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language task for the agent to perform.",
        min_length=1,
    )
    input_data: dict[str, Any] | None = Field(
        default=None,
        description="Optional context as key-value pairs (e.g. {'company': 'Acme', 'url': 'acme.com'}).",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the agent's response.",
    )


class RankInput(BaseModel):
    """Input for the rank operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language instructions for scoring a single row.",
        min_length=1,
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    field_name: str = Field(..., description="Name of the field to sort by.")
    field_type: Literal["float", "int", "str", "bool"] = Field(
        default="float",
        description="Type of the score field: 'float', 'int', 'str', or 'bool'",
    )
    ascending_order: bool = Field(
        default=True, description="Sort ascending (True) or descending (False)."
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "RankInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class ScreenInput(BaseModel):
    """Input for the screen operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ..., description="Natural language screening criteria.", min_length=1
    )
    input_csv: str | None = Field(
        default=None, description="Absolute path to the input CSV file."
    )
    input_data: str | None = Field(
        default=None,
        description="Raw CSV content as a string (alternative to input_csv for remote use).",
    )
    input_json: list[dict[str, Any]] | None = Field(
        default=None,
        description="Data as a JSON array of objects. "
        'Example: [{"company": "Acme", "url": "acme.com"}, {"company": "Beta", "url": "beta.io"}]',
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model. "
        "Must include at least one boolean property — screen uses the boolean field to filter rows into pass/fail.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_path(v)
        return v

    @model_validator(mode="after")
    def check_input_source(self) -> "ScreenInput":
        sources = sum(
            1 for s in (self.input_csv, self.input_data, self.input_json) if s
        )
        if sources == 0:
            raise ValueError("Provide one of input_csv, input_data, or input_json.")
        if sources > 1:
            raise ValueError(
                "Provide only one of input_csv, input_data, or input_json."
            )
        return self


class DedupeInput(BaseModel):
    """Input for the dedupe operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    equivalence_relation: str = Field(
        ...,
        description="Natural language description of what makes two rows equivalent/duplicates. "
        "The LLM will use this to identify which rows represent the same entity.",
        min_length=1,
    )
    input_csv: str = Field(..., description="Absolute path to the input CSV file.")

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class MergeInput(BaseModel):
    """Input for the merge operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of how to match rows.",
        min_length=1,
    )
    left_csv: str = Field(
        ...,
        description="Absolute path to the left CSV. Works like a LEFT JOIN: ALL rows from this table are kept in the output. This should be the table being enriched.",
    )
    right_csv: str = Field(
        ...,
        description="Absolute path to the right CSV. This is the lookup/reference table. Its columns are added to matching left rows; unmatched left rows get nulls.",
    )
    merge_on_left: str | None = Field(
        default=None,
        description="Only set if you expect some exact string matches on the chosen column or want to draw special attention of LLM agents to this particular column. Fine to leave unspecified in all other cases.",
    )
    merge_on_right: str | None = Field(
        default=None,
        description="Only set if you expect some exact string matches on the chosen column or want to draw special attention of LLM agents to this particular column. Fine to leave unspecified in all other cases.",
    )
    use_web_search: Literal["auto", "yes", "no"] | None = Field(
        default=None, description='Control web search: "auto", "yes", or "no".'
    )
    relationship_type: Literal["many_to_one", "one_to_one"] | None = Field(
        default=None,
        description="Leave unset for the default many_to_one, which is correct in most cases. many_to_one: multiple left rows can match one right row (e.g. products → companies). one_to_one: each left row matches at most one right row AND vice versa. Only use one_to_one when both tables represent unique entities of the same kind.",
    )

    @field_validator("left_csv", "right_csv")
    @classmethod
    def validate_csv_paths(cls, v: str) -> str:
        validate_csv_path(v)
        return v


class ProgressInput(BaseModel):
    """Input for checking task progress."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID returned by the operation tool.")


class ResultsInput(BaseModel):
    """Input for retrieving completed task results."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task_id: str = Field(..., description="The task ID of the completed task.")
    output_path: str | None = Field(
        default=None,
        description="Full absolute path to the output CSV file (must end in .csv). "
        "If omitted, results are returned inline.",
    )
    offset: int = Field(
        default=0,
        description="Row offset for pagination. Default 0 returns the first page.",
        ge=0,
    )
    page_size: int = Field(
        default=PREVIEW_SIZE,
        description="Number of rows per page. Default 5. Max 50.",
        ge=1,
        le=50,
    )

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str | None) -> str | None:
        if v is not None:
            validate_csv_output_path(v)
        return v
