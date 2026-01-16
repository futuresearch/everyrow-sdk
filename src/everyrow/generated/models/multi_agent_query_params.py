from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.llm_enum import LLMEnum
from ..models.multi_agent_effort_level import MultiAgentEffortLevel
from ..models.response_schema_type import ResponseSchemaType
from ..types import UNSET, Unset

T = TypeVar("T", bound="MultiAgentQueryParams")


@_attrs_define
class MultiAgentQueryParams:
    """
    Attributes:
        task (str): The task assigned to the multi-agent
        llm (LLMEnum | Unset):
        agenda (str | Unset): Research agenda, if any Default: ''.
        selected_input_fields (list[str] | None | Unset): Fields to select from the input artifacts. If None, select all
            fields.
        apply_at_depth (int | Unset):  Default: -1.
        preview (bool | Unset): When true, process only the first 5 inputs in map/map-expand operations Default: False.
        is_expand (bool | Unset): When true, treat outputs as lists (expand behavior) Default: False.
        flatten_map_expand (bool | Unset): When true with MAP mode and is_expand, flatten expanded results into single
            group instead of nested groups Default: True.
        response_schema (Any | Unset):  Default: {'_model_name': 'Root', 'answer': {'description': 'The response
            answer', 'optional': False, 'type': 'str'}}.
        response_schema_type (ResponseSchemaType | Unset): Type of response schema format.
        effort_level (MultiAgentEffortLevel | Unset): Effort level for the multi-agent task.
        agent_model (LLMEnum | Unset):
        driver_model (LLMEnum | Unset):
        agent_step_timeout_seconds (int | Unset): Agent step timeout in seconds (8-1400) Default: 300.
        timeout_seconds (int | Unset): Full task timeout in seconds (20-360000) Default: 600.
        retries (int | Unset): Number of retries (0-5) Default: 2.
    """

    task: str
    llm: LLMEnum | Unset = UNSET
    agenda: str | Unset = ""
    selected_input_fields: list[str] | None | Unset = UNSET
    apply_at_depth: int | Unset = -1
    preview: bool | Unset = False
    is_expand: bool | Unset = False
    flatten_map_expand: bool | Unset = True
    response_schema: Any | Unset = {
        "_model_name": "Root",
        "answer": {"description": "The response answer", "optional": False, "type": "str"},
    }
    response_schema_type: ResponseSchemaType | Unset = UNSET
    effort_level: MultiAgentEffortLevel | Unset = UNSET
    agent_model: LLMEnum | Unset = UNSET
    driver_model: LLMEnum | Unset = UNSET
    agent_step_timeout_seconds: int | Unset = 300
    timeout_seconds: int | Unset = 600
    retries: int | Unset = 2
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task = self.task

        llm: str | Unset = UNSET
        if not isinstance(self.llm, Unset):
            llm = self.llm.value

        agenda = self.agenda

        selected_input_fields: list[str] | None | Unset
        if isinstance(self.selected_input_fields, Unset):
            selected_input_fields = UNSET
        elif isinstance(self.selected_input_fields, list):
            selected_input_fields = self.selected_input_fields

        else:
            selected_input_fields = self.selected_input_fields

        apply_at_depth = self.apply_at_depth

        preview = self.preview

        is_expand = self.is_expand

        flatten_map_expand = self.flatten_map_expand

        response_schema = self.response_schema

        response_schema_type: str | Unset = UNSET
        if not isinstance(self.response_schema_type, Unset):
            response_schema_type = self.response_schema_type.value

        effort_level: str | Unset = UNSET
        if not isinstance(self.effort_level, Unset):
            effort_level = self.effort_level.value

        agent_model: str | Unset = UNSET
        if not isinstance(self.agent_model, Unset):
            agent_model = self.agent_model.value

        driver_model: str | Unset = UNSET
        if not isinstance(self.driver_model, Unset):
            driver_model = self.driver_model.value

        agent_step_timeout_seconds = self.agent_step_timeout_seconds

        timeout_seconds = self.timeout_seconds

        retries = self.retries

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task": task,
            }
        )
        if llm is not UNSET:
            field_dict["llm"] = llm
        if agenda is not UNSET:
            field_dict["agenda"] = agenda
        if selected_input_fields is not UNSET:
            field_dict["selected_input_fields"] = selected_input_fields
        if apply_at_depth is not UNSET:
            field_dict["apply_at_depth"] = apply_at_depth
        if preview is not UNSET:
            field_dict["preview"] = preview
        if is_expand is not UNSET:
            field_dict["is_expand"] = is_expand
        if flatten_map_expand is not UNSET:
            field_dict["flatten_map_expand"] = flatten_map_expand
        if response_schema is not UNSET:
            field_dict["response_schema"] = response_schema
        if response_schema_type is not UNSET:
            field_dict["response_schema_type"] = response_schema_type
        if effort_level is not UNSET:
            field_dict["effort_level"] = effort_level
        if agent_model is not UNSET:
            field_dict["agent_model"] = agent_model
        if driver_model is not UNSET:
            field_dict["driver_model"] = driver_model
        if agent_step_timeout_seconds is not UNSET:
            field_dict["agent_step_timeout_seconds"] = agent_step_timeout_seconds
        if timeout_seconds is not UNSET:
            field_dict["timeout_seconds"] = timeout_seconds
        if retries is not UNSET:
            field_dict["retries"] = retries

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task = d.pop("task")

        _llm = d.pop("llm", UNSET)
        llm: LLMEnum | Unset
        if isinstance(_llm, Unset):
            llm = UNSET
        else:
            llm = LLMEnum(_llm)

        agenda = d.pop("agenda", UNSET)

        def _parse_selected_input_fields(data: object) -> list[str] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                selected_input_fields_type_0 = cast(list[str], data)

                return selected_input_fields_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[str] | None | Unset, data)

        selected_input_fields = _parse_selected_input_fields(d.pop("selected_input_fields", UNSET))

        apply_at_depth = d.pop("apply_at_depth", UNSET)

        preview = d.pop("preview", UNSET)

        is_expand = d.pop("is_expand", UNSET)

        flatten_map_expand = d.pop("flatten_map_expand", UNSET)

        response_schema = d.pop("response_schema", UNSET)

        _response_schema_type = d.pop("response_schema_type", UNSET)
        response_schema_type: ResponseSchemaType | Unset
        if isinstance(_response_schema_type, Unset):
            response_schema_type = UNSET
        else:
            response_schema_type = ResponseSchemaType(_response_schema_type)

        _effort_level = d.pop("effort_level", UNSET)
        effort_level: MultiAgentEffortLevel | Unset
        if isinstance(_effort_level, Unset):
            effort_level = UNSET
        else:
            effort_level = MultiAgentEffortLevel(_effort_level)

        _agent_model = d.pop("agent_model", UNSET)
        agent_model: LLMEnum | Unset
        if isinstance(_agent_model, Unset):
            agent_model = UNSET
        else:
            agent_model = LLMEnum(_agent_model)

        _driver_model = d.pop("driver_model", UNSET)
        driver_model: LLMEnum | Unset
        if isinstance(_driver_model, Unset):
            driver_model = UNSET
        else:
            driver_model = LLMEnum(_driver_model)

        agent_step_timeout_seconds = d.pop("agent_step_timeout_seconds", UNSET)

        timeout_seconds = d.pop("timeout_seconds", UNSET)

        retries = d.pop("retries", UNSET)

        multi_agent_query_params = cls(
            task=task,
            llm=llm,
            agenda=agenda,
            selected_input_fields=selected_input_fields,
            apply_at_depth=apply_at_depth,
            preview=preview,
            is_expand=is_expand,
            flatten_map_expand=flatten_map_expand,
            response_schema=response_schema,
            response_schema_type=response_schema_type,
            effort_level=effort_level,
            agent_model=agent_model,
            driver_model=driver_model,
            agent_step_timeout_seconds=agent_step_timeout_seconds,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )

        multi_agent_query_params.additional_properties = d
        return multi_agent_query_params

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
