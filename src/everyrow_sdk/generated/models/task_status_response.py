from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.task_status import TaskStatus

T = TypeVar("T", bound="TaskStatusResponse")


@_attrs_define
class TaskStatusResponse:
    """Response containing task status and artifact information.

    Attributes:
        task_id (UUID):
        status (TaskStatus):
        artifact_id (None | UUID):
        error (None | str):
    """

    task_id: UUID
    status: TaskStatus
    artifact_id: None | UUID
    error: None | str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_id = str(self.task_id)

        status = self.status.value

        artifact_id: None | str
        if isinstance(self.artifact_id, UUID):
            artifact_id = str(self.artifact_id)
        else:
            artifact_id = self.artifact_id

        error: None | str
        error = self.error

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_id": task_id,
                "status": status,
                "artifact_id": artifact_id,
                "error": error,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task_id = UUID(d.pop("task_id"))

        status = TaskStatus(d.pop("status"))

        def _parse_artifact_id(data: object) -> None | UUID:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                artifact_id_type_0 = UUID(data)

                return artifact_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | UUID, data)

        artifact_id = _parse_artifact_id(d.pop("artifact_id"))

        def _parse_error(data: object) -> None | str:
            if data is None:
                return data
            return cast(None | str, data)

        error = _parse_error(d.pop("error"))

        task_status_response = cls(
            task_id=task_id,
            status=status,
            artifact_id=artifact_id,
            error=error,
        )

        task_status_response.additional_properties = d
        return task_status_response

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
