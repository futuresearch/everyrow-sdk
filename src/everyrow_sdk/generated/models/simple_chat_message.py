from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.simple_chat_message_role import SimpleChatMessageRole
from ..types import UNSET, Unset

T = TypeVar("T", bound="SimpleChatMessage")


@_attrs_define
class SimpleChatMessage:
    """
    Attributes:
        content (str):
        role (SimpleChatMessageRole):
        message_type (Literal['simple'] | Unset):  Default: 'simple'.
        completion_tokens (int | None | Unset):
        cacheable (bool | Unset):  Default: False.
        explicit_cache_breakpoint (bool | Unset):  Default: False.
    """

    content: str
    role: SimpleChatMessageRole
    message_type: Literal["simple"] | Unset = "simple"
    completion_tokens: int | None | Unset = UNSET
    cacheable: bool | Unset = False
    explicit_cache_breakpoint: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        content = self.content

        role = self.role.value

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
                "content": content,
                "role": role,
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
        d = dict(src_dict)
        content = d.pop("content")

        role = SimpleChatMessageRole(d.pop("role"))

        message_type = cast(Literal["simple"] | Unset, d.pop("message_type", UNSET))
        if message_type != "simple" and not isinstance(message_type, Unset):
            raise ValueError(f"message_type must match const 'simple', got '{message_type}'")

        def _parse_completion_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        completion_tokens = _parse_completion_tokens(d.pop("completion_tokens", UNSET))

        cacheable = d.pop("cacheable", UNSET)

        explicit_cache_breakpoint = d.pop("explicit_cache_breakpoint", UNSET)

        simple_chat_message = cls(
            content=content,
            role=role,
            message_type=message_type,
            completion_tokens=completion_tokens,
            cacheable=cacheable,
            explicit_cache_breakpoint=explicit_cache_breakpoint,
        )

        simple_chat_message.additional_properties = d
        return simple_chat_message

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
