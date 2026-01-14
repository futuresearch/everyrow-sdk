from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_improvement_instruction import AgentImprovementInstruction


T = TypeVar("T", bound="ControllerImprovementRound")


@_attrs_define
class ControllerImprovementRound:
    """
    Attributes:
        instructions (list[AgentImprovementInstruction] | Unset): Per-agent improvement instructions
    """

    instructions: list[AgentImprovementInstruction] | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        instructions: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.instructions, Unset):
            instructions = []
            for instructions_item_data in self.instructions:
                instructions_item = instructions_item_data.to_dict()
                instructions.append(instructions_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if instructions is not UNSET:
            field_dict["instructions"] = instructions

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_improvement_instruction import AgentImprovementInstruction

        d = dict(src_dict)
        _instructions = d.pop("instructions", UNSET)
        instructions: list[AgentImprovementInstruction] | Unset = UNSET
        if _instructions is not UNSET:
            instructions = []
            for instructions_item_data in _instructions:
                instructions_item = AgentImprovementInstruction.from_dict(instructions_item_data)

                instructions.append(instructions_item)

        controller_improvement_round = cls(
            instructions=instructions,
        )

        controller_improvement_round.additional_properties = d
        return controller_improvement_round

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
