from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.event_type import EventType
from ..types import UNSET, Unset

T = TypeVar("T", bound="ConversationChangedPayload")


@_attrs_define
class ConversationChangedPayload:
    """
    Attributes:
        conversation_ids (list[UUID]):
        event_type (EventType | Unset):
    """

    conversation_ids: list[UUID]
    event_type: EventType | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        conversation_ids = []
        for conversation_ids_item_data in self.conversation_ids:
            conversation_ids_item = str(conversation_ids_item_data)
            conversation_ids.append(conversation_ids_item)

        event_type: str | Unset = UNSET
        if not isinstance(self.event_type, Unset):
            event_type = self.event_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "conversation_ids": conversation_ids,
            }
        )
        if event_type is not UNSET:
            field_dict["event_type"] = event_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        conversation_ids = []
        _conversation_ids = d.pop("conversation_ids")
        for conversation_ids_item_data in _conversation_ids:
            conversation_ids_item = UUID(conversation_ids_item_data)

            conversation_ids.append(conversation_ids_item)

        _event_type = d.pop("event_type", UNSET)
        event_type: EventType | Unset
        if isinstance(_event_type, Unset):
            event_type = UNSET
        else:
            event_type = EventType(_event_type)

        conversation_changed_payload = cls(
            conversation_ids=conversation_ids,
            event_type=event_type,
        )

        conversation_changed_payload.additional_properties = d
        return conversation_changed_payload

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
