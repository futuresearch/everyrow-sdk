from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UnconditionalFraming")


@_attrs_define
class UnconditionalFraming:
    """
    Attributes:
        kind (Literal['unconditional'] | Unset):  Default: 'unconditional'.
    """

    kind: Literal["unconditional"] | Unset = "unconditional"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        kind = self.kind

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if kind is not UNSET:
            field_dict["kind"] = kind

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        kind = cast(Literal["unconditional"] | Unset, d.pop("kind", UNSET))
        if kind != "unconditional" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'unconditional', got '{kind}'")

        unconditional_framing = cls(
            kind=kind,
        )

        unconditional_framing.additional_properties = d
        return unconditional_framing

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
