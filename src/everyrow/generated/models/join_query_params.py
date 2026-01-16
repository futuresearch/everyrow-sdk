from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="JoinQueryParams")


@_attrs_define
class JoinQueryParams:
    """
    Attributes:
        join_fields (list[str]):
        auto_detect_join_fields (bool | Unset): When true, automatically detect all columns with matching values to join
            on. When enabled, join_fields is ignored. Default: False.
    """

    join_fields: list[str]
    auto_detect_join_fields: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        join_fields = self.join_fields

        auto_detect_join_fields = self.auto_detect_join_fields

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "join_fields": join_fields,
            }
        )
        if auto_detect_join_fields is not UNSET:
            field_dict["auto_detect_join_fields"] = auto_detect_join_fields

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        join_fields = cast(list[str], d.pop("join_fields"))

        auto_detect_join_fields = d.pop("auto_detect_join_fields", UNSET)

        join_query_params = cls(
            join_fields=join_fields,
            auto_detect_join_fields=auto_detect_join_fields,
        )

        join_query_params.additional_properties = d
        return join_query_params

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
