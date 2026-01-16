from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.status_count import StatusCount


T = TypeVar("T", bound="ProgressStatus")


@_attrs_define
class ProgressStatus:
    """
    Attributes:
        by_status (list[StatusCount]):
        total (int):
    """

    by_status: list[StatusCount]
    total: int
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        by_status = []
        for by_status_item_data in self.by_status:
            by_status_item = by_status_item_data.to_dict()
            by_status.append(by_status_item)

        total = self.total

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "by_status": by_status,
                "total": total,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.status_count import StatusCount

        d = dict(src_dict)
        by_status = []
        _by_status = d.pop("by_status")
        for by_status_item_data in _by_status:
            by_status_item = StatusCount.from_dict(by_status_item_data)

            by_status.append(by_status_item)

        total = d.pop("total")

        progress_status = cls(
            by_status=by_status,
            total=total,
        )

        progress_status.additional_properties = d
        return progress_status

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
