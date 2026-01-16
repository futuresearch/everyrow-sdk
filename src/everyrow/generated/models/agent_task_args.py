from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.agent_task_args_processing_mode import AgentTaskArgsProcessingMode
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.agent_query_params import AgentQueryParams


T = TypeVar("T", bound="AgentTaskArgs")


@_attrs_define
class AgentTaskArgs:
    """
    Attributes:
        processing_mode (AgentTaskArgsProcessingMode):
        query (AgentQueryParams):
        input_artifacts (list[UUID]):
        context_artifacts (list[UUID]):
        label (None | str | Unset): Very short label (a few words) clearly stating the action, e.g. 'Find third-party
            claims'.
        description (None | str | Unset): One to two sentence, high-level description of what this task does. If the
            output has many columns or parameters, do not list them all; give a holistic description instead.
        join_with_input (bool | Unset): Whether to include input columns alongside the generated columns in the output
            artifact. Only used for `map` processing mode. Default: True.
    """

    processing_mode: AgentTaskArgsProcessingMode
    query: AgentQueryParams
    input_artifacts: list[UUID]
    context_artifacts: list[UUID]
    label: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    join_with_input: bool | Unset = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        processing_mode = self.processing_mode.value

        query = self.query.to_dict()

        input_artifacts = []
        for componentsschemas_input_artifacts_ids_item_data in self.input_artifacts:
            componentsschemas_input_artifacts_ids_item = str(componentsschemas_input_artifacts_ids_item_data)
            input_artifacts.append(componentsschemas_input_artifacts_ids_item)

        context_artifacts = []
        for componentsschemas_context_artifacts_ids_item_data in self.context_artifacts:
            componentsschemas_context_artifacts_ids_item = str(componentsschemas_context_artifacts_ids_item_data)
            context_artifacts.append(componentsschemas_context_artifacts_ids_item)

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        description: None | str | Unset
        if isinstance(self.description, Unset):
            description = UNSET
        else:
            description = self.description

        join_with_input = self.join_with_input

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "processing_mode": processing_mode,
                "query": query,
                "input_artifacts": input_artifacts,
                "context_artifacts": context_artifacts,
            }
        )
        if label is not UNSET:
            field_dict["label"] = label
        if description is not UNSET:
            field_dict["description"] = description
        if join_with_input is not UNSET:
            field_dict["join_with_input"] = join_with_input

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.agent_query_params import AgentQueryParams

        d = dict(src_dict)
        processing_mode = AgentTaskArgsProcessingMode(d.pop("processing_mode"))

        query = AgentQueryParams.from_dict(d.pop("query"))

        input_artifacts = []
        _input_artifacts = d.pop("input_artifacts")
        for componentsschemas_input_artifacts_ids_item_data in _input_artifacts:
            componentsschemas_input_artifacts_ids_item = UUID(componentsschemas_input_artifacts_ids_item_data)

            input_artifacts.append(componentsschemas_input_artifacts_ids_item)

        context_artifacts = []
        _context_artifacts = d.pop("context_artifacts")
        for componentsschemas_context_artifacts_ids_item_data in _context_artifacts:
            componentsschemas_context_artifacts_ids_item = UUID(componentsschemas_context_artifacts_ids_item_data)

            context_artifacts.append(componentsschemas_context_artifacts_ids_item)

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_description(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        description = _parse_description(d.pop("description", UNSET))

        join_with_input = d.pop("join_with_input", UNSET)

        agent_task_args = cls(
            processing_mode=processing_mode,
            query=query,
            input_artifacts=input_artifacts,
            context_artifacts=context_artifacts,
            label=label,
            description=description,
            join_with_input=join_with_input,
        )

        agent_task_args.additional_properties = d
        return agent_task_args

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
