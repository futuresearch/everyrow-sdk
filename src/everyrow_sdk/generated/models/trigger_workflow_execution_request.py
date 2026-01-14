from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.trigger_workflow_execution_request_task_params import TriggerWorkflowExecutionRequestTaskParams
    from ..models.workflow_leaf_node_input import WorkflowLeafNodeInput


T = TypeVar("T", bound="TriggerWorkflowExecutionRequest")


@_attrs_define
class TriggerWorkflowExecutionRequest:
    """
    Attributes:
        workflow_id (UUID):
        inputs (list[WorkflowLeafNodeInput]):
        session_id (UUID):
        task_params (TriggerWorkflowExecutionRequestTaskParams | Unset):
    """

    workflow_id: UUID
    inputs: list[WorkflowLeafNodeInput]
    session_id: UUID
    task_params: TriggerWorkflowExecutionRequestTaskParams | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        workflow_id = str(self.workflow_id)

        inputs = []
        for inputs_item_data in self.inputs:
            inputs_item = inputs_item_data.to_dict()
            inputs.append(inputs_item)

        session_id = str(self.session_id)

        task_params: dict[str, Any] | Unset = UNSET
        if not isinstance(self.task_params, Unset):
            task_params = self.task_params.to_dict()

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "workflow_id": workflow_id,
                "inputs": inputs,
                "session_id": session_id,
            }
        )
        if task_params is not UNSET:
            field_dict["task_params"] = task_params

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.trigger_workflow_execution_request_task_params import TriggerWorkflowExecutionRequestTaskParams
        from ..models.workflow_leaf_node_input import WorkflowLeafNodeInput

        d = dict(src_dict)
        workflow_id = UUID(d.pop("workflow_id"))

        inputs = []
        _inputs = d.pop("inputs")
        for inputs_item_data in _inputs:
            inputs_item = WorkflowLeafNodeInput.from_dict(inputs_item_data)

            inputs.append(inputs_item)

        session_id = UUID(d.pop("session_id"))

        _task_params = d.pop("task_params", UNSET)
        task_params: TriggerWorkflowExecutionRequestTaskParams | Unset
        if isinstance(_task_params, Unset):
            task_params = UNSET
        else:
            task_params = TriggerWorkflowExecutionRequestTaskParams.from_dict(_task_params)

        trigger_workflow_execution_request = cls(
            workflow_id=workflow_id,
            inputs=inputs,
            session_id=session_id,
            task_params=task_params,
        )

        trigger_workflow_execution_request.additional_properties = d
        return trigger_workflow_execution_request

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
