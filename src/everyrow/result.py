from typing import TypeVar
from uuid import UUID

import attrs
from pandas import DataFrame
from pydantic import BaseModel

T = TypeVar("T", bound=str | BaseModel)


@attrs.define
class ScalarResult[T: str | BaseModel]:
    artifact_id: UUID
    data: T
    error: str | None


@attrs.define
class TableResult:
    artifact_id: UUID
    data: DataFrame
    error: str | None


Result = ScalarResult | TableResult
