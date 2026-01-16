from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.aux_data_source_bank import AuxDataSourceBank
    from ..models.date_cutoffs import DateCutoffs


T = TypeVar("T", bound="AuxData")


@_attrs_define
class AuxData:
    """
    Attributes:
        source_bank (AuxDataSourceBank | Unset):
        date_cutoffs (DateCutoffs | None | Unset):
        justification (None | str | Unset):
    """

    source_bank: AuxDataSourceBank | Unset = UNSET
    date_cutoffs: DateCutoffs | None | Unset = UNSET
    justification: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.date_cutoffs import DateCutoffs

        source_bank: dict[str, Any] | Unset = UNSET
        if not isinstance(self.source_bank, Unset):
            source_bank = self.source_bank.to_dict()

        date_cutoffs: dict[str, Any] | None | Unset
        if isinstance(self.date_cutoffs, Unset):
            date_cutoffs = UNSET
        elif isinstance(self.date_cutoffs, DateCutoffs):
            date_cutoffs = self.date_cutoffs.to_dict()
        else:
            date_cutoffs = self.date_cutoffs

        justification: None | str | Unset
        if isinstance(self.justification, Unset):
            justification = UNSET
        else:
            justification = self.justification

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if source_bank is not UNSET:
            field_dict["source_bank"] = source_bank
        if date_cutoffs is not UNSET:
            field_dict["date_cutoffs"] = date_cutoffs
        if justification is not UNSET:
            field_dict["justification"] = justification

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.aux_data_source_bank import AuxDataSourceBank
        from ..models.date_cutoffs import DateCutoffs

        d = dict(src_dict)
        _source_bank = d.pop("source_bank", UNSET)
        source_bank: AuxDataSourceBank | Unset
        if isinstance(_source_bank, Unset):
            source_bank = UNSET
        else:
            source_bank = AuxDataSourceBank.from_dict(_source_bank)

        def _parse_date_cutoffs(data: object) -> DateCutoffs | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                date_cutoffs_type_0 = DateCutoffs.from_dict(data)

                return date_cutoffs_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(DateCutoffs | None | Unset, data)

        date_cutoffs = _parse_date_cutoffs(d.pop("date_cutoffs", UNSET))

        def _parse_justification(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        justification = _parse_justification(d.pop("justification", UNSET))

        aux_data = cls(
            source_bank=source_bank,
            date_cutoffs=date_cutoffs,
            justification=justification,
        )

        aux_data.additional_properties = d
        return aux_data

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
