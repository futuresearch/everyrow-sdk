from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="ExecutionMetadata")


@_attrs_define
class ExecutionMetadata:
    """
    Attributes:
        subtasks (list[UUID]):
        final_task (UUID):
        final_task_on_error (UUID):
        chord (UUID):
        orchestrator_task (None | Unset | UUID):
        orchestrator_controller_id (None | str | Unset):
    """

    subtasks: list[UUID]
    final_task: UUID
    final_task_on_error: UUID
    chord: UUID
    orchestrator_task: None | Unset | UUID = UNSET
    orchestrator_controller_id: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        subtasks = []
        for subtasks_item_data in self.subtasks:
            subtasks_item = str(subtasks_item_data)
            subtasks.append(subtasks_item)

        final_task = str(self.final_task)

        final_task_on_error = str(self.final_task_on_error)

        chord = str(self.chord)

        orchestrator_task: None | str | Unset
        if isinstance(self.orchestrator_task, Unset):
            orchestrator_task = UNSET
        elif isinstance(self.orchestrator_task, UUID):
            orchestrator_task = str(self.orchestrator_task)
        else:
            orchestrator_task = self.orchestrator_task

        orchestrator_controller_id: None | str | Unset
        if isinstance(self.orchestrator_controller_id, Unset):
            orchestrator_controller_id = UNSET
        else:
            orchestrator_controller_id = self.orchestrator_controller_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "subtasks": subtasks,
                "final_task": final_task,
                "final_task_on_error": final_task_on_error,
                "chord": chord,
            }
        )
        if orchestrator_task is not UNSET:
            field_dict["orchestrator_task"] = orchestrator_task
        if orchestrator_controller_id is not UNSET:
            field_dict["orchestrator_controller_id"] = orchestrator_controller_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        subtasks = []
        _subtasks = d.pop("subtasks")
        for subtasks_item_data in _subtasks:
            subtasks_item = UUID(subtasks_item_data)

            subtasks.append(subtasks_item)

        final_task = UUID(d.pop("final_task"))

        final_task_on_error = UUID(d.pop("final_task_on_error"))

        chord = UUID(d.pop("chord"))

        def _parse_orchestrator_task(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                orchestrator_task_type_0 = UUID(data)

                return orchestrator_task_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        orchestrator_task = _parse_orchestrator_task(d.pop("orchestrator_task", UNSET))

        def _parse_orchestrator_controller_id(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        orchestrator_controller_id = _parse_orchestrator_controller_id(d.pop("orchestrator_controller_id", UNSET))

        execution_metadata = cls(
            subtasks=subtasks,
            final_task=final_task,
            final_task_on_error=final_task_on_error,
            chord=chord,
            orchestrator_task=orchestrator_task,
            orchestrator_controller_id=orchestrator_controller_id,
        )

        execution_metadata.additional_properties = d
        return execution_metadata

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
