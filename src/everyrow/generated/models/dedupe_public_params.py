from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define

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

    def to_dict(self) -> dict[str, Any]:
        equivalence_relation = self.equivalence_relation

        select_representative = self.select_representative

        preview = self.preview

        field_dict: dict[str, Any] = {}

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

        return dedupe_public_params
