from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="UsageResponse")


@_attrs_define
class UsageResponse:
    """
    Attributes:
        usage (int):
        capacity (int):
        distinct_tasks (int):
    """

    usage: int
    capacity: int
    distinct_tasks: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        usage = self.usage

        capacity = self.capacity

        distinct_tasks = self.distinct_tasks

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "usage": usage,
                "capacity": capacity,
                "distinct_tasks": distinct_tasks,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        usage = d.pop("usage")

        capacity = d.pop("capacity")

        distinct_tasks = d.pop("distinct_tasks")

        usage_response = cls(
            usage=usage,
            capacity=capacity,
            distinct_tasks=distinct_tasks,
        )

        usage_response.additional_properties = d
        return usage_response

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
