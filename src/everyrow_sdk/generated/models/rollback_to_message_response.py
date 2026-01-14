from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="RollbackToMessageResponse")


@_attrs_define
class RollbackToMessageResponse:
    """
    Attributes:
        deleted_messages (int):
        deleted_tasks (int):
        message (str):
    """

    deleted_messages: int
    deleted_tasks: int
    message: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        deleted_messages = self.deleted_messages

        deleted_tasks = self.deleted_tasks

        message = self.message

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "deleted_messages": deleted_messages,
                "deleted_tasks": deleted_tasks,
                "message": message,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        deleted_messages = d.pop("deleted_messages")

        deleted_tasks = d.pop("deleted_tasks")

        message = d.pop("message")

        rollback_to_message_response = cls(
            deleted_messages=deleted_messages,
            deleted_tasks=deleted_tasks,
            message=message,
        )

        rollback_to_message_response.additional_properties = d
        return rollback_to_message_response

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
