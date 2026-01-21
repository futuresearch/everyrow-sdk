from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DedupePublicParams")


@_attrs_define
class DedupePublicParams:
    """Public-facing parameters for the deduplication service.

    Attributes:
        equivalence_relation (str): Description of what makes items equivalent
        select_representative (bool | Unset): When true, use LLM to select the best representative from each equivalence
            class. When false, no selection is made. Default: True.
        preview (bool | Unset):  Default: False.
    """

    equivalence_relation: str
    select_representative: bool | Unset = True
    preview: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        equivalence_relation = self.equivalence_relation

        select_representative = self.select_representative

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "equivalence_relation": equivalence_relation,
            }
        )
        if select_representative is not UNSET:
            field_dict["select_representative"] = select_representative
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        equivalence_relation = d.pop("equivalence_relation")

        select_representative = d.pop("select_representative", UNSET)

        preview = d.pop("preview", UNSET)

        dedupe_public_params = cls(
            equivalence_relation=equivalence_relation,
            select_representative=select_representative,
            preview=preview,
        )

        dedupe_public_params.additional_properties = d
        return dedupe_public_params

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
