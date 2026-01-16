from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.chat_completion_message_tool_call import ChatCompletionMessageToolCall


T = TypeVar("T", bound="SimpleChatMessageWithToolCalls")


@_attrs_define
class SimpleChatMessageWithToolCalls:
    """
    Attributes:
        role (Literal['assistant']):
        tool_calls (list[ChatCompletionMessageToolCall]):
        message_type (Literal['simple_with_tool_calls'] | Unset):  Default: 'simple_with_tool_calls'.
        content (None | str | Unset):
        completion_tokens (int | None | Unset):
        cacheable (bool | Unset):  Default: False.
        explicit_cache_breakpoint (bool | Unset):  Default: False.
    """

    role: Literal["assistant"]
    tool_calls: list[ChatCompletionMessageToolCall]
    message_type: Literal["simple_with_tool_calls"] | Unset = "simple_with_tool_calls"
    content: None | str | Unset = UNSET
    completion_tokens: int | None | Unset = UNSET
    cacheable: bool | Unset = False
    explicit_cache_breakpoint: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        role = self.role

        tool_calls = []
        for tool_calls_item_data in self.tool_calls:
            tool_calls_item = tool_calls_item_data.to_dict()
            tool_calls.append(tool_calls_item)

        message_type = self.message_type

        content: None | str | Unset
        if isinstance(self.content, Unset):
            content = UNSET
        else:
            content = self.content

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
                "tool_calls": tool_calls,
            }
        )
        if message_type is not UNSET:
            field_dict["message_type"] = message_type
        if content is not UNSET:
            field_dict["content"] = content
        if completion_tokens is not UNSET:
            field_dict["completion_tokens"] = completion_tokens
        if cacheable is not UNSET:
            field_dict["cacheable"] = cacheable
        if explicit_cache_breakpoint is not UNSET:
            field_dict["explicit_cache_breakpoint"] = explicit_cache_breakpoint

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.chat_completion_message_tool_call import ChatCompletionMessageToolCall

        d = dict(src_dict)
        role = cast(Literal["assistant"], d.pop("role"))
        if role != "assistant":
            raise ValueError(f"role must match const 'assistant', got '{role}'")

        tool_calls = []
        _tool_calls = d.pop("tool_calls")
        for tool_calls_item_data in _tool_calls:
            tool_calls_item = ChatCompletionMessageToolCall.from_dict(tool_calls_item_data)

            tool_calls.append(tool_calls_item)

        message_type = cast(Literal["simple_with_tool_calls"] | Unset, d.pop("message_type", UNSET))
        if message_type != "simple_with_tool_calls" and not isinstance(message_type, Unset):
            raise ValueError(f"message_type must match const 'simple_with_tool_calls', got '{message_type}'")

        def _parse_content(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        content = _parse_content(d.pop("content", UNSET))

        def _parse_completion_tokens(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        completion_tokens = _parse_completion_tokens(d.pop("completion_tokens", UNSET))

        cacheable = d.pop("cacheable", UNSET)

        explicit_cache_breakpoint = d.pop("explicit_cache_breakpoint", UNSET)

        simple_chat_message_with_tool_calls = cls(
            role=role,
            tool_calls=tool_calls,
            message_type=message_type,
            content=content,
            completion_tokens=completion_tokens,
            cacheable=cacheable,
            explicit_cache_breakpoint=explicit_cache_breakpoint,
        )

        simple_chat_message_with_tool_calls.additional_properties = d
        return simple_chat_message_with_tool_calls

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
