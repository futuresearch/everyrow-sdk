from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="PreviewMetadata")


@_attrs_define
class PreviewMetadata:
    """
    Attributes:
        original_task_id (UUID):
        index (int | None | Unset):
        label (None | str | Unset):
        description (None | str | Unset):
        num_omitted_artifacts (int | None | Unset):
    """

    original_task_id: UUID
    index: int | None | Unset = UNSET
    label: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    num_omitted_artifacts: int | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        original_task_id = str(self.original_task_id)

        index: int | None | Unset
        if isinstance(self.index, Unset):
            index = UNSET
        else:
            index = self.index

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        num_omitted_artifacts: int | None | Unset
        if isinstance(self.num_omitted_artifacts, Unset):
            num_omitted_artifacts = UNSET
        else:
            num_omitted_artifacts = self.num_omitted_artifacts

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "original_task_id": original_task_id,
            }
        )
        if index is not UNSET:
            field_dict["index"] = index
        if label is not UNSET:
            field_dict["label"] = label
        if description is not UNSET:
            field_dict["description"] = description
        if num_omitted_artifacts is not UNSET:
            field_dict["num_omitted_artifacts"] = num_omitted_artifacts

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        original_task_id = UUID(d.pop("original_task_id"))

        def _parse_index(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        index = _parse_index(d.pop("index", UNSET))

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_num_omitted_artifacts(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        num_omitted_artifacts = _parse_num_omitted_artifacts(d.pop("num_omitted_artifacts", UNSET))

        preview_metadata = cls(
            original_task_id=original_task_id,
            index=index,
            label=label,
            description=description,
            num_omitted_artifacts=num_omitted_artifacts,
        )

        preview_metadata.additional_properties = d
        return preview_metadata

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
