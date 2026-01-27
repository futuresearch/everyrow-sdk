from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="ResourceEstimationResponse")


@_attrs_define
class ResourceEstimationResponse:
    """
    Attributes:
        cost_lower (float): Lower estimate on the cost to complete the task in USD
        cost_upper (float): Upper estimate on the cost to complete the task in USD
        time_lower (float): Lower estimate on the time to complete the task in seconds
        time_upper (float): Upper estimate on the time to complete the task in seconds
    """

    cost_lower: float
    cost_upper: float
    time_lower: float
    time_upper: float
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        cost_lower = self.cost_lower

        cost_upper = self.cost_upper

        time_lower = self.time_lower

        time_upper = self.time_upper

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "cost_lower": cost_lower,
                "cost_upper": cost_upper,
                "time_lower": time_lower,
                "time_upper": time_upper,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        cost_lower = d.pop("cost_lower")

        cost_upper = d.pop("cost_upper")

        time_lower = d.pop("time_lower")

        time_upper = d.pop("time_upper")

        resource_estimation_response = cls(
            cost_lower=cost_lower,
            cost_upper=cost_upper,
            time_lower=time_lower,
            time_upper=time_upper,
        )

        resource_estimation_response.additional_properties = d
        return resource_estimation_response

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
