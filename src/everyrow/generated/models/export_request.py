from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.export_request_token_data import ExportRequestTokenData


T = TypeVar("T", bound="ExportRequest")


@_attrs_define
class ExportRequest:
    """
    Attributes:
        token_data (ExportRequestTokenData):
        artifact_id (str):
    """

    token_data: ExportRequestTokenData
    artifact_id: str
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        token_data = self.token_data.to_dict()

        artifact_id = self.artifact_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "token_data": token_data,
                "artifact_id": artifact_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.export_request_token_data import ExportRequestTokenData

        d = dict(src_dict)
        token_data = ExportRequestTokenData.from_dict(d.pop("token_data"))

        artifact_id = d.pop("artifact_id")

        export_request = cls(
            token_data=token_data,
            artifact_id=artifact_id,
        )

        export_request.additional_properties = d
        return export_request

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
