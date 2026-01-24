"""MCP server for everyrow SDK operations."""

import json
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from everyrow.api_utils import create_client
from everyrow.generated.api.default.whoami_whoami_get import asyncio as whoami
from everyrow.ops import agent_map, dedupe, merge, rank, screen
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from everyrow_mcp.utils import (
    resolve_output_path,
    save_result_to_csv,
    validate_csv_path,
    validate_output_path,
)


@asynccontextmanager
async def lifespan(_server: FastMCP):
    """Validate everyrow credentials on startup."""
    try:
        client = create_client()
        async with client as c:
            response = await whoami(client=c)
            if response is None:
                raise RuntimeError("Failed to authenticate with everyrow API")
        print("everyrow-mcp: Authenticated successfully")
    except Exception as e:
        raise RuntimeError(f"everyrow-mcp startup failed: {e}") from e

    yield


mcp = FastMCP("everyrow_mcp", lifespan=lifespan)


class ScreenInput(BaseModel):
    """Input for the screen operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of the screening criteria. "
        "Rows that meet the criteria will pass the screen.",
        min_length=1,
    )
    input_csv: str = Field(
        ...,
        description="Absolute path to the input CSV file to screen.",
    )
    output_path: str = Field(
        ...,
        description="Output path: either a directory (file will be named 'screened_<input_name>.csv') "
        "or a full file path ending in .csv",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model. "
        "If not provided, uses a default schema with a 'passes' boolean field. "
        "The schema should define fields that the LLM will extract for each row.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_output_path(v)
        return v


@mcp.tool(name="everyrow_screen")
async def everyrow_screen(params: ScreenInput) -> str:
    """Filter rows in a CSV based on criteria that require judgment.

    Screen evaluates each row against natural language criteria and keeps
    only rows that pass. Useful for filtering based on semantic meaning
    rather than exact string matching.

    Examples:
    - Filter job postings for "remote-friendly AND senior-level AND salary disclosed"
    - Screen vendors for "financially stable AND good security practices"
    - Filter leads for "likely to need our product based on company description"

    Args:
        params: ScreenInput containing task, input_csv path, output_path, and optional response_schema

    Returns:
        JSON string with result summary including output file path and row counts
    """
    df = pd.read_csv(params.input_csv)
    input_rows = len(df)

    response_model = None
    if params.response_schema:
        response_model = _schema_to_model("ScreenResult", params.response_schema)

    result = await screen(
        task=params.task,
        input=df,
        response_model=response_model,
    )

    output_file = resolve_output_path(params.output_path, params.input_csv, "screened")
    save_result_to_csv(result.data, output_file)

    return json.dumps(
        {
            "status": "success",
            "output_file": str(output_file),
            "input_rows": input_rows,
            "output_rows": len(result.data),
            "rows_filtered": input_rows - len(result.data),
        },
        indent=2,
    )


class RankInput(BaseModel):
    """Input for the rank operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of the ranking criteria. "
        "Describes what makes a row score higher or lower.",
        min_length=1,
    )
    input_csv: str = Field(
        ...,
        description="Absolute path to the input CSV file to rank.",
    )
    output_path: str = Field(
        ...,
        description="Output path: either a directory (file will be named 'ranked_<input_name>.csv') "
        "or a full file path ending in .csv",
    )
    field_name: str = Field(
        ...,
        description="Name of the field to use for sorting. "
        "This field will be added to the output with the LLM-assigned scores.",
    )
    field_type: str = Field(
        default="float",
        description="Type of the ranking field: 'float', 'int', 'str', or 'bool'",
    )
    ascending_order: bool = Field(
        default=True,
        description="If True, sort in ascending order (lowest first). "
        "If False, sort in descending order (highest first).",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema for the response model. "
        "Must include the field_name as a property. "
        "If not provided, a simple schema with just field_name is used.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_output_path(v)
        return v

    @field_validator("field_type")
    @classmethod
    def validate_field_type(cls, v: str) -> str:
        valid_types = {"float", "int", "str", "bool"}
        if v not in valid_types:
            raise ValueError(f"field_type must be one of {valid_types}")
        return v


@mcp.tool(name="everyrow_rank")
async def everyrow_rank(params: RankInput) -> str:
    """Score and sort rows in a CSV based on qualitative criteria.

    Rank evaluates each row and assigns a score based on the task description,
    then sorts the table by that score. Useful for prioritizing items based
    on semantic evaluation.

    Examples:
    - Rank leads by "likelihood to need data integration solutions"
    - Sort companies by "AI/ML adoption maturity"
    - Prioritize candidates by "fit for senior engineering role"

    Args:
        params: RankInput containing task, input_csv, output_path, field_name, and options

    Returns:
        JSON string with result summary including output file path
    """
    df = pd.read_csv(params.input_csv)

    response_model = None
    if params.response_schema:
        response_model = _schema_to_model("RankResult", params.response_schema)

    result = await rank(
        task=params.task,
        input=df,
        field_name=params.field_name,
        field_type=params.field_type,  # type: ignore
        response_model=response_model,
        ascending_order=params.ascending_order,
    )

    output_file = resolve_output_path(params.output_path, params.input_csv, "ranked")
    save_result_to_csv(result.data, output_file)

    return json.dumps(
        {
            "status": "success",
            "output_file": str(output_file),
            "rows": len(result.data),
            "sorted_by": params.field_name,
            "ascending": params.ascending_order,
        },
        indent=2,
    )


class DedupeInput(BaseModel):
    """Input for the dedupe operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    equivalence_relation: str = Field(
        ...,
        description="Natural language description of what makes two rows equivalent/duplicates. "
        "The LLM will use this to identify which rows represent the same entity.",
        min_length=1,
    )
    input_csv: str = Field(
        ...,
        description="Absolute path to the input CSV file to deduplicate.",
    )
    output_path: str = Field(
        ...,
        description="Output path: either a directory (file will be named 'deduped_<input_name>.csv') "
        "or a full file path ending in .csv",
    )
    select_representative: bool = Field(
        default=True,
        description="If True, select one representative row per duplicate group. "
        "If False, keep all rows but mark duplicates with equivalence class info.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_output_path(v)
        return v


@mcp.tool(name="everyrow_dedupe")
async def everyrow_dedupe(params: DedupeInput) -> str:
    """Remove duplicate rows from a CSV using semantic equivalence.

    Dedupe identifies rows that represent the same entity even when they
    don't match exactly. Useful for fuzzy deduplication where string
    matching fails.

    Examples:
    - Dedupe contacts: "Same person even with name abbreviations or career changes"
    - Dedupe companies: "Same company including subsidiaries and name variations"
    - Dedupe research papers: "Same work including preprints and published versions"

    Args:
        params: DedupeInput containing equivalence_relation, input_csv, output_path, and options

    Returns:
        JSON string with result summary including output file path and dedup stats
    """
    df = pd.read_csv(params.input_csv)
    input_rows = len(df)

    result = await dedupe(
        equivalence_relation=params.equivalence_relation,
        input=df,
        select_representative=params.select_representative,
    )

    output_file = resolve_output_path(params.output_path, params.input_csv, "deduped")
    save_result_to_csv(result.data, output_file)

    return json.dumps(
        {
            "status": "success",
            "output_file": str(output_file),
            "input_rows": input_rows,
            "output_rows": len(result.data),
            "duplicates_removed": input_rows - len(result.data),
        },
        indent=2,
    )


class MergeInput(BaseModel):
    """Input for the merge operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of how to match rows between the two tables. "
        "Describes the relationship between entities in left and right tables.",
        min_length=1,
    )
    left_csv: str = Field(
        ...,
        description="Absolute path to the left/primary CSV file.",
    )
    right_csv: str = Field(
        ...,
        description="Absolute path to the right/secondary CSV file to merge in.",
    )
    output_path: str = Field(
        ...,
        description="Output path: either a directory (file will be named 'merged_<left_name>.csv') "
        "or a full file path ending in .csv",
    )
    merge_on_left: str | None = Field(
        default=None,
        description="Optional column name in the left table to use as the merge key. "
        "If not provided, the LLM will determine the best matching strategy.",
    )
    merge_on_right: str | None = Field(
        default=None,
        description="Optional column name in the right table to use as the merge key. "
        "If not provided, the LLM will determine the best matching strategy.",
    )

    @field_validator("left_csv", "right_csv")
    @classmethod
    def validate_csv_paths(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_output_path(v)
        return v


@mcp.tool(name="everyrow_merge")
async def everyrow_merge(params: MergeInput) -> str:
    """Join two CSV files using intelligent entity matching.

    Merge combines two tables even when keys don't match exactly. The LLM
    performs research and reasoning to identify which rows should be joined.

    Examples:
    - Match software products to parent companies (Photoshop -> Adobe)
    - Match clinical trial sponsors to pharma companies (Genentech -> Roche)
    - Join contact lists with different name formats

    Args:
        params: MergeInput containing task, left_csv, right_csv, output_path, and optional merge keys

    Returns:
        JSON string with result summary including output file path and merge stats
    """
    left_df = pd.read_csv(params.left_csv)
    right_df = pd.read_csv(params.right_csv)

    result = await merge(
        task=params.task,
        left_table=left_df,
        right_table=right_df,
        merge_on_left=params.merge_on_left,
        merge_on_right=params.merge_on_right,
    )

    output_file = resolve_output_path(params.output_path, params.left_csv, "merged")
    save_result_to_csv(result.data, output_file)

    return json.dumps(
        {
            "status": "success",
            "output_file": str(output_file),
            "left_rows": len(left_df),
            "right_rows": len(right_df),
            "output_rows": len(result.data),
        },
        indent=2,
    )


class AgentInput(BaseModel):
    """Input for the agent operation."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(
        ...,
        description="Natural language description of the task to perform on each row. "
        "The agent will execute this task independently for each row in the input.",
        min_length=1,
    )
    input_csv: str = Field(
        ...,
        description="Absolute path to the input CSV file. The agent will process each row.",
    )
    output_path: str = Field(
        ...,
        description="Output path: either a directory (file will be named 'agent_<input_name>.csv') "
        "or a full file path ending in .csv",
    )
    response_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional JSON schema defining the structure of the agent's response. "
        "If not provided, uses a default schema with an 'answer' string field. "
        "The schema defines what fields the agent should extract/generate for each row.",
    )

    @field_validator("input_csv")
    @classmethod
    def validate_input_csv(cls, v: str) -> str:
        validate_csv_path(v)
        return v

    @field_validator("output_path")
    @classmethod
    def validate_output(cls, v: str) -> str:
        validate_output_path(v)
        return v


@mcp.tool(name="everyrow_agent")
async def everyrow_agent(params: AgentInput) -> str:
    """Run web research agents on each row of a CSV.

    Agent performs web research and extraction tasks on each row independently.
    Useful for enriching data with information from the web.

    Examples:
    - "Find this company's latest funding round and lead investors"
    - "Research the CEO's background and previous companies"
    - "Find pricing information for this product"

    Args:
        params: AgentInput containing task, input_csv, output_path, and optional response_schema

    Returns:
        JSON string with result summary including output file path
    """
    df = pd.read_csv(params.input_csv)

    if params.response_schema:
        response_model = _schema_to_model("AgentResult", params.response_schema)
        result = await agent_map(
            task=params.task,
            input=df,
            response_model=response_model,
        )
    else:
        result = await agent_map(
            task=params.task,
            input=df,
        )

    output_file = resolve_output_path(params.output_path, params.input_csv, "agent")
    save_result_to_csv(result.data, output_file)

    return json.dumps(
        {
            "status": "success",
            "output_file": str(output_file),
            "rows_processed": len(result.data),
        },
        indent=2,
    )


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


def main():
    """Run the MCP server."""
    # Check for API key before starting
    if "EVERYROW_API_KEY" not in os.environ:
        print(
            "Error: EVERYROW_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        print(
            "Get an API key at https://everyrow.io/api-key",
            file=sys.stderr,
        )
        sys.exit(1)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
