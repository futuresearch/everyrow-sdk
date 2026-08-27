from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DecisionFraming")


@_attrs_define
class DecisionFraming:
    """
    Attributes:
        alternatives_field (str): Input column whose cell holds the row's alternatives
        kind (Literal['decision'] | Unset):  Default: 'decision'.
        intervention (None | str | Unset): What executing an alternative is assumed to mean. Null applies the default
            assumptions.
    """

    alternatives_field: str
    kind: Literal["decision"] | Unset = "decision"
    intervention: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        alternatives_field = self.alternatives_field

        kind = self.kind

        intervention: None | str | Unset
        if isinstance(self.intervention, Unset):
            intervention = UNSET
        else:
            intervention = self.intervention

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "alternatives_field": alternatives_field,
            }
        )
        if kind is not UNSET:
            field_dict["kind"] = kind
        if intervention is not UNSET:
            field_dict["intervention"] = intervention

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        alternatives_field = d.pop("alternatives_field")

        kind = cast(Literal["decision"] | Unset, d.pop("kind", UNSET))
        if kind != "decision" and not isinstance(kind, Unset):
            raise ValueError(f"kind must match const 'decision', got '{kind}'")

        def _parse_intervention(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        intervention = _parse_intervention(d.pop("intervention", UNSET))

        decision_framing = cls(
            alternatives_field=alternatives_field,
            kind=kind,
            intervention=intervention,
        )

        decision_framing.additional_properties = d
        return decision_framing

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
