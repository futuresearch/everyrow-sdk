from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.allowed_suggestions import AllowedSuggestions
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.preview_metadata import PreviewMetadata


T = TypeVar("T", bound="ChatMessageMetadata")


@_attrs_define
class ChatMessageMetadata:
    """
    Attributes:
        task_id (None | str | Unset | UUID):
        preview (None | PreviewMetadata | Unset):
        tool_name (None | str | Unset):
        description (None | str | Unset):
        suggestion (AllowedSuggestions | None | Unset):
    """

    task_id: None | str | Unset | UUID = UNSET
    preview: None | PreviewMetadata | Unset = UNSET
    tool_name: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    suggestion: AllowedSuggestions | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.preview_metadata import PreviewMetadata

        task_id: None | str | Unset
        if isinstance(self.task_id, Unset):
            task_id = UNSET
        elif isinstance(self.task_id, UUID):
            task_id = str(self.task_id)
        else:
            task_id = self.task_id

        preview: dict[str, Any] | None | Unset
        if isinstance(self.preview, Unset):
            preview = UNSET
        elif isinstance(self.preview, PreviewMetadata):
            preview = self.preview.to_dict()
        else:
            preview = self.preview

        tool_name: None | str | Unset
        if isinstance(self.tool_name, Unset):
            tool_name = UNSET
        else:
            tool_name = self.tool_name

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        suggestion: None | str | Unset
        if isinstance(self.suggestion, Unset):
            suggestion = UNSET
        elif isinstance(self.suggestion, AllowedSuggestions):
            suggestion = self.suggestion.value
        else:
            suggestion = self.suggestion

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if task_id is not UNSET:
            field_dict["task_id"] = task_id
        if preview is not UNSET:
            field_dict["preview"] = preview
        if tool_name is not UNSET:
            field_dict["tool_name"] = tool_name
        if description is not UNSET:
            field_dict["description"] = description
        if suggestion is not UNSET:
            field_dict["suggestion"] = suggestion

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.preview_metadata import PreviewMetadata

        d = dict(src_dict)

        def _parse_task_id(data: object) -> None | str | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                task_id_type_0 = UUID(data)

                return task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | str | Unset | UUID, data)

        task_id = _parse_task_id(d.pop("task_id", UNSET))

        def _parse_preview(data: object) -> None | PreviewMetadata | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                preview_type_0 = PreviewMetadata.from_dict(data)

                return preview_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | PreviewMetadata | Unset, data)

        preview = _parse_preview(d.pop("preview", UNSET))

        def _parse_tool_name(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        tool_name = _parse_tool_name(d.pop("tool_name", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        def _parse_suggestion(data: object) -> AllowedSuggestions | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                suggestion_type_0 = AllowedSuggestions(data)

                return suggestion_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AllowedSuggestions | None | Unset, data)

        suggestion = _parse_suggestion(d.pop("suggestion", UNSET))

        chat_message_metadata = cls(
            task_id=task_id,
            preview=preview,
            tool_name=tool_name,
            description=description,
            suggestion=suggestion,
        )

        chat_message_metadata.additional_properties = d
        return chat_message_metadata

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
