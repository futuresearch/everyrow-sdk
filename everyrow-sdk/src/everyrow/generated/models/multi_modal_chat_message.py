from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.multi_modal_chat_message_role import MultiModalChatMessageRole
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.image_chat_content_part import ImageChatContentPart
    from ..models.text_chat_content_part import TextChatContentPart


T = TypeVar("T", bound="MultiModalChatMessage")


@_attrs_define
class MultiModalChatMessage:
    """
    Attributes:
        role (MultiModalChatMessageRole):
        content (list[ImageChatContentPart | TextChatContentPart]):
        message_type (Literal['multi_modal'] | Unset):  Default: 'multi_modal'.
        completion_tokens (int | None | Unset):
        cacheable (bool | Unset):  Default: False.
        explicit_cache_breakpoint (bool | Unset):  Default: False.
    """

    role: MultiModalChatMessageRole
    content: list[ImageChatContentPart | TextChatContentPart]
    message_type: Literal["multi_modal"] | Unset = "multi_modal"
    completion_tokens: int | None | Unset = UNSET
    cacheable: bool | Unset = False
    explicit_cache_breakpoint: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.text_chat_content_part import TextChatContentPart

        role = self.role.value

        content = []
        for content_item_data in self.content:
            content_item: dict[str, Any]
            if isinstance(content_item_data, TextChatContentPart):
                content_item = content_item_data.to_dict()
            else:
                content_item = content_item_data.to_dict()

            content.append(content_item)

        message_type = self.message_type

        completion_tokens: int | None | Unset
        if isinstance(self.completion_tokens, Unset):
            completion_tokens = UNSET
        else:
            completion_tokens = self.completion_tokens

        cacheable = self.cacheable

        explicit_cache_breakpoint = self.explicit_cache_breakpoint

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "role": role,
                "content": content,
            }
        )
        if message_type is not UNSET:
            field_dict["message_type"] = message_type
        if completion_tokens is not UNSET:
            field_dict["completion_tokens"] = completion_tokens
        if cacheable is not UNSET:
            field_dict["cacheable"] = cacheable
        if explicit_cache_breakpoint is not UNSET:
            field_dict["explicit_cache_breakpoint"] = explicit_cache_breakpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.image_chat_content_part import ImageChatContentPart
        from ..models.text_chat_content_part import TextChatContentPart

        d = dict(src_dict)
        role = MultiModalChatMessageRole(d.pop("role"))

        content = []
        _content = d.pop("content")
        for content_item_data in _content:

            def _parse_content_item(data: object) -> ImageChatContentPart | TextChatContentPart:
                try:
                    if not isinstance(data, dict):
                        raise TypeError()
                    content_item_type_0 = TextChatContentPart.from_dict(data)

                    return content_item_type_0
                except (TypeError, ValueError, AttributeError, KeyError):
                    pass
                if not isinstance(data, dict):
                    raise TypeError()
                content_item_type_1 = ImageChatContentPart.from_dict(data)

                return content_item_type_1

            content_item = _parse_content_item(content_item_data)

            content.append(content_item)

        message_type = cast(Literal["multi_modal"] | Unset, d.pop("message_type", UNSET))
        if message_type != "multi_modal" and not isinstance(message_type, Unset):
            raise ValueError(f"message_type must match const 'multi_modal', got '{message_type}'")

        def _parse_completion_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        completion_tokens = _parse_completion_tokens(d.pop("completion_tokens", UNSET))

        cacheable = d.pop("cacheable", UNSET)

        explicit_cache_breakpoint = d.pop("explicit_cache_breakpoint", UNSET)

        multi_modal_chat_message = cls(
            role=role,
            content=content,
            message_type=message_type,
            completion_tokens=completion_tokens,
            cacheable=cacheable,
            explicit_cache_breakpoint=explicit_cache_breakpoint,
        )

        multi_modal_chat_message.additional_properties = d
        return multi_modal_chat_message

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
