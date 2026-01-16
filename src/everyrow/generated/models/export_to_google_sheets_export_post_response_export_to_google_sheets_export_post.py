from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost")


@_attrs_define
class ExportToGoogleSheetsExportPostResponseExportToGoogleSheetsExportPost:
    """ """

    additional_properties: dict[str, str] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        export_to_google_sheets_export_post_response_export_to_google_sheets_export_post = cls()

        export_to_google_sheets_export_post_response_export_to_google_sheets_export_post.additional_properties = d
        return export_to_google_sheets_export_post_response_export_to_google_sheets_export_post

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> str:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: str) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
