from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolResponseMessage")


@_attrs_define
class ToolResponseMessage:
    """
    Attributes:
        tool_call_id (str):
        content (str):
        description (str):
        tool_name (str):
        message_type (Literal['tool_response'] | Unset):  Default: 'tool_response'.
        role (Literal['tool'] | Unset):  Default: 'tool'.
        cacheable (bool | Unset):  Default: False.
        explicit_cache_breakpoint (bool | Unset):  Default: False.
    """

    tool_call_id: str
    content: str
    description: str
    tool_name: str
    message_type: Literal["tool_response"] | Unset = "tool_response"
    role: Literal["tool"] | Unset = "tool"
    cacheable: bool | Unset = False
    explicit_cache_breakpoint: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        tool_call_id = self.tool_call_id

        content = self.content

        description = self.description

        tool_name = self.tool_name

        message_type = self.message_type

        role = self.role

        cacheable = self.cacheable

        explicit_cache_breakpoint = self.explicit_cache_breakpoint

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "tool_call_id": tool_call_id,
                "content": content,
                "description": description,
                "tool_name": tool_name,
            }
        )
        if message_type is not UNSET:
            field_dict["message_type"] = message_type
        if role is not UNSET:
            field_dict["role"] = role
        if cacheable is not UNSET:
            field_dict["cacheable"] = cacheable
        if explicit_cache_breakpoint is not UNSET:
            field_dict["explicit_cache_breakpoint"] = explicit_cache_breakpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        tool_call_id = d.pop("tool_call_id")

        content = d.pop("content")

        description = d.pop("description")

        tool_name = d.pop("tool_name")

        message_type = cast(Literal["tool_response"] | Unset, d.pop("message_type", UNSET))
        if message_type != "tool_response" and not isinstance(message_type, Unset):
            raise ValueError(f"message_type must match const 'tool_response', got '{message_type}'")

        role = cast(Literal["tool"] | Unset, d.pop("role", UNSET))
        if role != "tool" and not isinstance(role, Unset):
            raise ValueError(f"role must match const 'tool', got '{role}'")

        cacheable = d.pop("cacheable", UNSET)

        explicit_cache_breakpoint = d.pop("explicit_cache_breakpoint", UNSET)

        tool_response_message = cls(
            tool_call_id=tool_call_id,
            content=content,
            description=description,
            tool_name=tool_name,
            message_type=message_type,
            role=role,
            cacheable=cacheable,
            explicit_cache_breakpoint=explicit_cache_breakpoint,
        )

        tool_response_message.additional_properties = d
        return tool_response_message

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
