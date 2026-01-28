from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

import attrs
from pandas import DataFrame

if TYPE_CHECKING:
    from pydantic import BaseModel

T = TypeVar("T", bound="str | BaseModel | dict[str, Any]")


@attrs.define
class ScalarResult[T: "str | BaseModel | dict[str, Any]"]:
    artifact_id: UUID
    data: T
    error: str | None


@attrs.define
class TableResult:
    artifact_id: UUID
    data: DataFrame
    error: str | None


Result = ScalarResult | TableResult
