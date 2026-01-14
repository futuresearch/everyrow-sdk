from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="UploadCsvQueryParams")


@_attrs_define
class UploadCsvQueryParams:
    """
    Attributes:
        file (str): Base64 encoded CSV file
        filename (None | str | Unset): Original filename of the uploaded CSV
        group_artifact_id (None | Unset | UUID): Optional group artifact ID to use for the uploaded data
    """

    file: str
    filename: None | str | Unset = UNSET
    group_artifact_id: None | Unset | UUID = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        file = self.file

        filename: None | str | Unset
        if isinstance(self.filename, Unset):
            filename = UNSET
        else:
            filename = self.filename

        group_artifact_id: None | str | Unset
        if isinstance(self.group_artifact_id, Unset):
            group_artifact_id = UNSET
        elif isinstance(self.group_artifact_id, UUID):
            group_artifact_id = str(self.group_artifact_id)
        else:
            group_artifact_id = self.group_artifact_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "file": file,
            }
        )
        if filename is not UNSET:
            field_dict["filename"] = filename
        if group_artifact_id is not UNSET:
            field_dict["group_artifact_id"] = group_artifact_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        file = d.pop("file")

        def _parse_filename(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        filename = _parse_filename(d.pop("filename", UNSET))

        def _parse_group_artifact_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                group_artifact_id_type_0 = UUID(data)

                return group_artifact_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        group_artifact_id = _parse_group_artifact_id(d.pop("group_artifact_id", UNSET))

        upload_csv_query_params = cls(
            file=file,
            filename=filename,
            group_artifact_id=group_artifact_id,
        )

        upload_csv_query_params.additional_properties = d
        return upload_csv_query_params

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
