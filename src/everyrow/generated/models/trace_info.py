from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="TraceInfo")


@_attrs_define
class TraceInfo:
    """
    Attributes:
        task_id (UUID):
        trace_id (UUID):
        artifact_id (UUID):
    """

    task_id: UUID
    trace_id: UUID
    artifact_id: UUID
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_id = str(self.task_id)

        trace_id = str(self.trace_id)

        artifact_id = str(self.artifact_id)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_id": task_id,
                "trace_id": trace_id,
                "artifact_id": artifact_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task_id = UUID(d.pop("task_id"))

        trace_id = UUID(d.pop("trace_id"))

        artifact_id = UUID(d.pop("artifact_id"))

        trace_info = cls(
            task_id=task_id,
            trace_id=trace_id,
            artifact_id=artifact_id,
        )

        trace_info.additional_properties = d
        return trace_info

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
