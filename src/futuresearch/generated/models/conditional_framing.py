from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ConditionalFraming")


@_attrs_define
class ConditionalFraming:
    """
    Attributes:
        kind (Literal['conditional'] | Unset):  Default: 'conditional'.
        condition (None | str | Unset): The condition, as a single question shared by every row
        condition_field (None | str | Unset): Input column whose cell holds each row's own condition
    """

    kind: Literal["conditional"] | Unset = "conditional"
    condition: None | str | Unset = UNSET
    condition_field: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind

        condition: None | str | Unset
        if isinstance(self.condition, Unset):
            condition = UNSET
        else:
            condition = self.condition

        condition_field: None | str | Unset
        if isinstance(self.condition_field, Unset):
            condition_field = UNSET
        else:
            condition_field = self.condition_field

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if kind is not UNSET:
            field_dict["kind"] = kind
        if condition is not UNSET:
            field_dict["condition"] = condition
        if condition_field is not UNSET:
            field_dict["condition_field"] = condition_field

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = cast(Literal["conditional"] | Unset, d.pop("kind", UNSET))
        if kind != "conditional" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'conditional', got '{kind}'")

        def _parse_condition(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        condition = _parse_condition(d.pop("condition", UNSET))

        def _parse_condition_field(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        condition_field = _parse_condition_field(d.pop("condition_field", UNSET))

        conditional_framing = cls(
            kind=kind,
            condition=condition,
            condition_field=condition_field,
        )

        conditional_framing.additional_properties = d
        return conditional_framing

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
