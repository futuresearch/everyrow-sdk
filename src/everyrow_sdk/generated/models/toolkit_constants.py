from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ToolkitConstants")


@_attrs_define
class ToolkitConstants:
    """
    Attributes:
        low_effort_agent_iteration_budget (Literal[5] | Unset):  Default: 5.
        high_effort_agent_iteration_budget (Literal[10] | Unset):  Default: 10.
    """

    low_effort_agent_iteration_budget: Literal[5] | Unset = 5
    high_effort_agent_iteration_budget: Literal[10] | Unset = 10
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        low_effort_agent_iteration_budget = self.low_effort_agent_iteration_budget

        high_effort_agent_iteration_budget = self.high_effort_agent_iteration_budget

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if low_effort_agent_iteration_budget is not UNSET:
            field_dict["low_effort_agent_iteration_budget"] = low_effort_agent_iteration_budget
        if high_effort_agent_iteration_budget is not UNSET:
            field_dict["high_effort_agent_iteration_budget"] = high_effort_agent_iteration_budget

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        low_effort_agent_iteration_budget = cast(Literal[5] | Unset, d.pop("low_effort_agent_iteration_budget", UNSET))
        if low_effort_agent_iteration_budget != 5 and not isinstance(low_effort_agent_iteration_budget, Unset):
            raise ValueError(
                f"low_effort_agent_iteration_budget must match const 5, got '{low_effort_agent_iteration_budget}'"
            )

        high_effort_agent_iteration_budget = cast(
            Literal[10] | Unset, d.pop("high_effort_agent_iteration_budget", UNSET)
        )
        if high_effort_agent_iteration_budget != 10 and not isinstance(high_effort_agent_iteration_budget, Unset):
            raise ValueError(
                f"high_effort_agent_iteration_budget must match const 10, got '{high_effort_agent_iteration_budget}'"
            )

        toolkit_constants = cls(
            low_effort_agent_iteration_budget=low_effort_agent_iteration_budget,
            high_effort_agent_iteration_budget=high_effort_agent_iteration_budget,
        )

        toolkit_constants.additional_properties = d
        return toolkit_constants

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
