from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.agent_query_params_system_prompt_kind_type_0 import AgentQueryParamsSystemPromptKindType0
from ..models.document_query_tool import DocumentQueryTool
from ..models.llm_enum import LLMEnum
from ..models.response_schema_type import ResponseSchemaType
from ..models.task_effort import TaskEffort
from ..types import UNSET, Unset

T = TypeVar("T", bound="AgentQueryParams")


@_attrs_define
class AgentQueryParams:
    """
    Attributes:
        task (str): The user-specified task
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
        number_of_steps (int | None | Unset): The number of agent iterations allowed to complete the task, after which
            the agent is forced to provide an answer, if it has not already done so. If not specified, defaults based on
            effort_level.
        generate_assumptions (bool | Unset):  Default: False.
        include_provenance_and_notes (bool | Unset): Whether to include an additional provenance and notes field in the
            output and prompt Default: True.
        agent_step_timeout_seconds (int | Unset): Agent step timeout in seconds (8-1400) Default: 300.
        timeout_seconds (int | Unset): Full task timeout in seconds (20-360000) Default: 600.
        retries (int | Unset): Number of retries (0-5) Default: 2.
        enable_communication (bool | Unset): Whether to enable communication between agents Default: False.
        max_feedback_rounds (int | Unset): Maximum number of rounds in orchestrator v2 mode.1 is equivalent to a
            standard map task. 2 is map -> feedback -> map Default: 1.
        document_query_tool (DocumentQueryTool | Unset): Document query tool to use.
        effort_level (TaskEffort | Unset): Effort level for task execution (includes MINIMAL for non-agent execution).
        system_prompt_kind (AgentQueryParamsSystemPromptKindType0 | None | Unset): System prompt variant to use. If not
            specified, defaults based on effort_level (speed_focused for LOW, persistent for HIGH)
        batch_size (int | None | Unset): The number of artifacts to process in a single batch. If set, enables batched
            agent processing.
        minimal_llm_system_prompt (None | str | Unset): Custom system prompt for minimal LLM calls (number_of_steps=0).
            If not provided, a default prompt with today's date is used.
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
    number_of_steps: int | None | Unset = UNSET
    generate_assumptions: bool | Unset = False
    include_provenance_and_notes: bool | Unset = True
    agent_step_timeout_seconds: int | Unset = 300
    timeout_seconds: int | Unset = 600
    retries: int | Unset = 2
    enable_communication: bool | Unset = False
    max_feedback_rounds: int | Unset = 1
    document_query_tool: DocumentQueryTool | Unset = UNSET
    effort_level: TaskEffort | Unset = UNSET
    system_prompt_kind: AgentQueryParamsSystemPromptKindType0 | None | Unset = UNSET
    batch_size: int | None | Unset = UNSET
    minimal_llm_system_prompt: None | str | Unset = UNSET
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

        number_of_steps: int | None | Unset
        if isinstance(self.number_of_steps, Unset):
            number_of_steps = UNSET
        else:
            number_of_steps = self.number_of_steps

        generate_assumptions = self.generate_assumptions

        include_provenance_and_notes = self.include_provenance_and_notes

        agent_step_timeout_seconds = self.agent_step_timeout_seconds

        timeout_seconds = self.timeout_seconds

        retries = self.retries

        enable_communication = self.enable_communication

        max_feedback_rounds = self.max_feedback_rounds

        document_query_tool: str | Unset = UNSET
        if not isinstance(self.document_query_tool, Unset):
            document_query_tool = self.document_query_tool.value

        effort_level: str | Unset = UNSET
        if not isinstance(self.effort_level, Unset):
            effort_level = self.effort_level.value

        system_prompt_kind: None | str | Unset
        if isinstance(self.system_prompt_kind, Unset):
            system_prompt_kind = UNSET
        elif isinstance(self.system_prompt_kind, AgentQueryParamsSystemPromptKindType0):
            system_prompt_kind = self.system_prompt_kind.value
        else:
            system_prompt_kind = self.system_prompt_kind

        batch_size: int | None | Unset
        if isinstance(self.batch_size, Unset):
            batch_size = UNSET
        else:
            batch_size = self.batch_size

        minimal_llm_system_prompt: None | str | Unset
        if isinstance(self.minimal_llm_system_prompt, Unset):
            minimal_llm_system_prompt = UNSET
        else:
            minimal_llm_system_prompt = self.minimal_llm_system_prompt

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
        if number_of_steps is not UNSET:
            field_dict["number_of_steps"] = number_of_steps
        if generate_assumptions is not UNSET:
            field_dict["generate_assumptions"] = generate_assumptions
        if include_provenance_and_notes is not UNSET:
            field_dict["include_provenance_and_notes"] = include_provenance_and_notes
        if agent_step_timeout_seconds is not UNSET:
            field_dict["agent_step_timeout_seconds"] = agent_step_timeout_seconds
        if timeout_seconds is not UNSET:
            field_dict["timeout_seconds"] = timeout_seconds
        if retries is not UNSET:
            field_dict["retries"] = retries
        if enable_communication is not UNSET:
            field_dict["enable_communication"] = enable_communication
        if max_feedback_rounds is not UNSET:
            field_dict["max_feedback_rounds"] = max_feedback_rounds
        if document_query_tool is not UNSET:
            field_dict["document_query_tool"] = document_query_tool
        if effort_level is not UNSET:
            field_dict["effort_level"] = effort_level
        if system_prompt_kind is not UNSET:
            field_dict["system_prompt_kind"] = system_prompt_kind
        if batch_size is not UNSET:
            field_dict["batch_size"] = batch_size
        if minimal_llm_system_prompt is not UNSET:
            field_dict["minimal_llm_system_prompt"] = minimal_llm_system_prompt

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

        def _parse_number_of_steps(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        number_of_steps = _parse_number_of_steps(d.pop("number_of_steps", UNSET))

        generate_assumptions = d.pop("generate_assumptions", UNSET)

        include_provenance_and_notes = d.pop("include_provenance_and_notes", UNSET)

        agent_step_timeout_seconds = d.pop("agent_step_timeout_seconds", UNSET)

        timeout_seconds = d.pop("timeout_seconds", UNSET)

        retries = d.pop("retries", UNSET)

        enable_communication = d.pop("enable_communication", UNSET)

        max_feedback_rounds = d.pop("max_feedback_rounds", UNSET)

        _document_query_tool = d.pop("document_query_tool", UNSET)
        document_query_tool: DocumentQueryTool | Unset
        if isinstance(_document_query_tool, Unset):
            document_query_tool = UNSET
        else:
            document_query_tool = DocumentQueryTool(_document_query_tool)

        _effort_level = d.pop("effort_level", UNSET)
        effort_level: TaskEffort | Unset
        if isinstance(_effort_level, Unset):
            effort_level = UNSET
        else:
            effort_level = TaskEffort(_effort_level)

        def _parse_system_prompt_kind(data: object) -> AgentQueryParamsSystemPromptKindType0 | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                system_prompt_kind_type_0 = AgentQueryParamsSystemPromptKindType0(data)

                return system_prompt_kind_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(AgentQueryParamsSystemPromptKindType0 | None | Unset, data)

        system_prompt_kind = _parse_system_prompt_kind(d.pop("system_prompt_kind", UNSET))

        def _parse_batch_size(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        batch_size = _parse_batch_size(d.pop("batch_size", UNSET))

        def _parse_minimal_llm_system_prompt(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        minimal_llm_system_prompt = _parse_minimal_llm_system_prompt(d.pop("minimal_llm_system_prompt", UNSET))

        agent_query_params = cls(
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
            number_of_steps=number_of_steps,
            generate_assumptions=generate_assumptions,
            include_provenance_and_notes=include_provenance_and_notes,
            agent_step_timeout_seconds=agent_step_timeout_seconds,
            timeout_seconds=timeout_seconds,
            retries=retries,
            enable_communication=enable_communication,
            max_feedback_rounds=max_feedback_rounds,
            document_query_tool=document_query_tool,
            effort_level=effort_level,
            system_prompt_kind=system_prompt_kind,
            batch_size=batch_size,
            minimal_llm_system_prompt=minimal_llm_system_prompt,
        )

        agent_query_params.additional_properties = d
        return agent_query_params

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
