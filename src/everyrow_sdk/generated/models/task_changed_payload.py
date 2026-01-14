from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.event_type import EventType
from ..types import UNSET, Unset

T = TypeVar("T", bound="TaskChangedPayload")


@_attrs_define
class TaskChangedPayload:
    """
    Attributes:
        task_ids (list[UUID]):
        artifact_ids (list[UUID]):
        event_type (EventType | Unset):
    """

    task_ids: list[UUID]
    artifact_ids: list[UUID]
    event_type: EventType | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task_ids = []
        for task_ids_item_data in self.task_ids:
            task_ids_item = str(task_ids_item_data)
            task_ids.append(task_ids_item)

        artifact_ids = []
        for artifact_ids_item_data in self.artifact_ids:
            artifact_ids_item = str(artifact_ids_item_data)
            artifact_ids.append(artifact_ids_item)

        event_type: str | Unset = UNSET
        if not isinstance(self.event_type, Unset):
            event_type = self.event_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_ids": task_ids,
                "artifact_ids": artifact_ids,
            }
        )
        if event_type is not UNSET:
            field_dict["event_type"] = event_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task_ids = []
        _task_ids = d.pop("task_ids")
        for task_ids_item_data in _task_ids:
            task_ids_item = UUID(task_ids_item_data)

            task_ids.append(task_ids_item)

        artifact_ids = []
        _artifact_ids = d.pop("artifact_ids")
        for artifact_ids_item_data in _artifact_ids:
            artifact_ids_item = UUID(artifact_ids_item_data)

            artifact_ids.append(artifact_ids_item)

        _event_type = d.pop("event_type", UNSET)
        event_type: EventType | Unset
        if isinstance(_event_type, Unset):
            event_type = UNSET
        else:
            event_type = EventType(_event_type)

        task_changed_payload = cls(
            task_ids=task_ids,
            artifact_ids=artifact_ids,
            event_type=event_type,
        )

        task_changed_payload.additional_properties = d
        return task_changed_payload

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
