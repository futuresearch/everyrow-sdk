from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.llm_enum import LLMEnum
from ..types import UNSET, Unset

T = TypeVar("T", bound="DedupePublicParams")


@_attrs_define
class DedupePublicParams:
    """Public-facing parameters for the deduplication service.

    Attributes:
        equivalence_relation (str): Description of what makes items equivalent
        llm (LLMEnum | Unset):
        chunk_size (int | Unset): Maximum number of items to process in a single LLM call Default: 25.
        use_clustering (bool | Unset): When true, cluster items by embedding similarity and only compare neighboring
            clusters. When false, use sequential chunking and compare all chunks (O(n²)) Default: True.
        select_representative (bool | Unset): When true, use LLM to select the best representative from each equivalence
            class. When false, no selection is made. Default: True.
        early_stopping_threshold (int | None | Unset): Stop cross-chunk comparisons for a row after this many
            consecutive comparisons with no matches. None disables early stopping. Default: 5.
        preview (bool | Unset):  Default: False.
    """

    equivalence_relation: str
    llm: LLMEnum | Unset = UNSET
    chunk_size: int | Unset = 25
    use_clustering: bool | Unset = True
    select_representative: bool | Unset = True
    early_stopping_threshold: int | None | Unset = 5
    preview: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        equivalence_relation = self.equivalence_relation

        llm: str | Unset = UNSET
        if not isinstance(self.llm, Unset):
            llm = self.llm.value

        chunk_size = self.chunk_size

        use_clustering = self.use_clustering

        select_representative = self.select_representative

        early_stopping_threshold: int | None | Unset
        if isinstance(self.early_stopping_threshold, Unset):
            early_stopping_threshold = UNSET
        else:
            early_stopping_threshold = self.early_stopping_threshold

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "equivalence_relation": equivalence_relation,
            }
        )
        if llm is not UNSET:
            field_dict["llm"] = llm
        if chunk_size is not UNSET:
            field_dict["chunk_size"] = chunk_size
        if use_clustering is not UNSET:
            field_dict["use_clustering"] = use_clustering
        if select_representative is not UNSET:
            field_dict["select_representative"] = select_representative
        if early_stopping_threshold is not UNSET:
            field_dict["early_stopping_threshold"] = early_stopping_threshold
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        equivalence_relation = d.pop("equivalence_relation")

        _llm = d.pop("llm", UNSET)
        llm: LLMEnum | Unset
        if isinstance(_llm, Unset):
            llm = UNSET
        else:
            llm = LLMEnum(_llm)

        chunk_size = d.pop("chunk_size", UNSET)

        use_clustering = d.pop("use_clustering", UNSET)

        select_representative = d.pop("select_representative", UNSET)

        def _parse_early_stopping_threshold(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        early_stopping_threshold = _parse_early_stopping_threshold(d.pop("early_stopping_threshold", UNSET))

        preview = d.pop("preview", UNSET)

        dedupe_public_params = cls(
            equivalence_relation=equivalence_relation,
            llm=llm,
            chunk_size=chunk_size,
            use_clustering=use_clustering,
            select_representative=select_representative,
            early_stopping_threshold=early_stopping_threshold,
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
