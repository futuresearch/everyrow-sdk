from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.event_type import EventType
from ..types import UNSET, Unset

T = TypeVar("T", bound="MessageCreatedPayload")


@_attrs_define
class MessageCreatedPayload:
    """
    Attributes:
        conversation_ids (list[UUID]):
        event_type (EventType | Unset):
        hide_on_frontend (bool | Unset):  Default: False.
    """

    conversation_ids: list[UUID]
    event_type: EventType | Unset = UNSET
    hide_on_frontend: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        conversation_ids = []
        for conversation_ids_item_data in self.conversation_ids:
            conversation_ids_item = str(conversation_ids_item_data)
            conversation_ids.append(conversation_ids_item)

        event_type: str | Unset = UNSET
        if not isinstance(self.event_type, Unset):
            event_type = self.event_type.value

        hide_on_frontend = self.hide_on_frontend

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "conversation_ids": conversation_ids,
            }
        )
        if event_type is not UNSET:
            field_dict["event_type"] = event_type
        if hide_on_frontend is not UNSET:
            field_dict["hide_on_frontend"] = hide_on_frontend

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

        hide_on_frontend = d.pop("hide_on_frontend", UNSET)

        message_created_payload = cls(
            conversation_ids=conversation_ids,
            event_type=event_type,
            hide_on_frontend=hide_on_frontend,
        )

        message_created_payload.additional_properties = d
        return message_created_payload

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
