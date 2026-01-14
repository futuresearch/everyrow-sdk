from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.event_type import EventType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.trace_info import TraceInfo


T = TypeVar("T", bound="TraceChangedPayload")


@_attrs_define
class TraceChangedPayload:
    """
    Attributes:
        traces (list[TraceInfo]):
        event_type (EventType | Unset):
    """

    traces: list[TraceInfo]
    event_type: EventType | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        traces = []
        for traces_item_data in self.traces:
            traces_item = traces_item_data.to_dict()
            traces.append(traces_item)

        event_type: str | Unset = UNSET
        if not isinstance(self.event_type, Unset):
            event_type = self.event_type.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "traces": traces,
            }
        )
        if event_type is not UNSET:
            field_dict["event_type"] = event_type

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trace_info import TraceInfo

        d = dict(src_dict)
        traces = []
        _traces = d.pop("traces")
        for traces_item_data in _traces:
            traces_item = TraceInfo.from_dict(traces_item_data)

            traces.append(traces_item)

        _event_type = d.pop("event_type", UNSET)
        event_type: EventType | Unset
        if isinstance(_event_type, Unset):
            event_type = UNSET
        else:
            event_type = EventType(_event_type)

        trace_changed_payload = cls(
            traces=traces,
            event_type=event_type,
        )

        trace_changed_payload.additional_properties = d
        return trace_changed_payload

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
