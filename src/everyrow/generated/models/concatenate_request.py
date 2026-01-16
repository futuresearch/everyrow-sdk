from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.concatenate_query_params import ConcatenateQueryParams


T = TypeVar("T", bound="ConcatenateRequest")


@_attrs_define
class ConcatenateRequest:
    """Request to concatenate multiple artifacts into a single group.

    Attributes:
        query (ConcatenateQueryParams): No specific parameters needed for concatenation - just combines all input
            artifacts
        input_artifacts (list[UUID] | None | Unset):
        context_artifacts (list[UUID] | None | Unset):
        label (None | str | Unset): Short task label for use in the UI
        description (None | str | Unset): Task description for use in the UI
        task_id (None | Unset | UUID):
        replaces_task_id (None | Unset | UUID): The ID of the task that this task replaces. Used e.g. by the full
            version of a task that replaces a preview version.
        twin_artifact_id (None | Unset | UUID): The ID of a reference artifact, e.g. the right table in Deep Merge
            operation.
        task_type (Literal['concatenate'] | Unset):  Default: 'concatenate'.
        processing_mode (Literal['transform'] | Unset):  Default: 'transform'.
    """

    query: ConcatenateQueryParams
    input_artifacts: list[UUID] | None | Unset = UNSET
    context_artifacts: list[UUID] | None | Unset = UNSET
    label: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    task_id: None | Unset | UUID = UNSET
    replaces_task_id: None | Unset | UUID = UNSET
    twin_artifact_id: None | Unset | UUID = UNSET
    task_type: Literal["concatenate"] | Unset = "concatenate"
    processing_mode: Literal["transform"] | Unset = "transform"
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        query = self.query.to_dict()

        input_artifacts: list[str] | None | Unset
        if isinstance(self.input_artifacts, Unset):
            input_artifacts = UNSET
        elif isinstance(self.input_artifacts, list):
            input_artifacts = []
            for componentsschemas_input_artifacts_ids_item_data in self.input_artifacts:
                componentsschemas_input_artifacts_ids_item = str(componentsschemas_input_artifacts_ids_item_data)
                input_artifacts.append(componentsschemas_input_artifacts_ids_item)

        else:
            input_artifacts = self.input_artifacts

        context_artifacts: list[str] | None | Unset
        if isinstance(self.context_artifacts, Unset):
            context_artifacts = UNSET
        elif isinstance(self.context_artifacts, list):
            context_artifacts = []
            for componentsschemas_context_artifacts_ids_item_data in self.context_artifacts:
                componentsschemas_context_artifacts_ids_item = str(componentsschemas_context_artifacts_ids_item_data)
                context_artifacts.append(componentsschemas_context_artifacts_ids_item)

        else:
            context_artifacts = self.context_artifacts

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

        task_id: None | str | Unset
        if isinstance(self.task_id, Unset):
            task_id = UNSET
        elif isinstance(self.task_id, UUID):
            task_id = str(self.task_id)
        else:
            task_id = self.task_id

        replaces_task_id: None | str | Unset
        if isinstance(self.replaces_task_id, Unset):
            replaces_task_id = UNSET
        elif isinstance(self.replaces_task_id, UUID):
            replaces_task_id = str(self.replaces_task_id)
        else:
            replaces_task_id = self.replaces_task_id

        twin_artifact_id: None | str | Unset
        if isinstance(self.twin_artifact_id, Unset):
            twin_artifact_id = UNSET
        elif isinstance(self.twin_artifact_id, UUID):
            twin_artifact_id = str(self.twin_artifact_id)
        else:
            twin_artifact_id = self.twin_artifact_id

        task_type = self.task_type

        processing_mode = self.processing_mode

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "query": query,
            }
        )
        if input_artifacts is not UNSET:
            field_dict["input_artifacts"] = input_artifacts
        if context_artifacts is not UNSET:
            field_dict["context_artifacts"] = context_artifacts
        if label is not UNSET:
            field_dict["label"] = label
        if description is not UNSET:
            field_dict["description"] = description
        if task_id is not UNSET:
            field_dict["task_id"] = task_id
        if replaces_task_id is not UNSET:
            field_dict["replaces_task_id"] = replaces_task_id
        if twin_artifact_id is not UNSET:
            field_dict["twin_artifact_id"] = twin_artifact_id
        if task_type is not UNSET:
            field_dict["task_type"] = task_type
        if processing_mode is not UNSET:
            field_dict["processing_mode"] = processing_mode

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.concatenate_query_params import ConcatenateQueryParams

        d = dict(src_dict)
        query = ConcatenateQueryParams.from_dict(d.pop("query"))

        def _parse_input_artifacts(data: object) -> list[UUID] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                input_artifacts_type_0 = []
                _input_artifacts_type_0 = data
                for componentsschemas_input_artifacts_ids_item_data in _input_artifacts_type_0:
                    componentsschemas_input_artifacts_ids_item = UUID(componentsschemas_input_artifacts_ids_item_data)

                    input_artifacts_type_0.append(componentsschemas_input_artifacts_ids_item)

                return input_artifacts_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[UUID] | None | Unset, data)

        input_artifacts = _parse_input_artifacts(d.pop("input_artifacts", UNSET))

        def _parse_context_artifacts(data: object) -> list[UUID] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                context_artifacts_type_0 = []
                _context_artifacts_type_0 = data
                for componentsschemas_context_artifacts_ids_item_data in _context_artifacts_type_0:
                    componentsschemas_context_artifacts_ids_item = UUID(
                        componentsschemas_context_artifacts_ids_item_data
                    )

                    context_artifacts_type_0.append(componentsschemas_context_artifacts_ids_item)

                return context_artifacts_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[UUID] | None | Unset, data)

        context_artifacts = _parse_context_artifacts(d.pop("context_artifacts", UNSET))

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

        def _parse_task_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                task_id_type_0 = UUID(data)

                return task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        task_id = _parse_task_id(d.pop("task_id", UNSET))

        def _parse_replaces_task_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                replaces_task_id_type_0 = UUID(data)

                return replaces_task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        replaces_task_id = _parse_replaces_task_id(d.pop("replaces_task_id", UNSET))

        def _parse_twin_artifact_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                twin_artifact_id_type_0 = UUID(data)

                return twin_artifact_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        twin_artifact_id = _parse_twin_artifact_id(d.pop("twin_artifact_id", UNSET))

        task_type = cast(Literal["concatenate"] | Unset, d.pop("task_type", UNSET))
        if task_type != "concatenate" and not isinstance(task_type, Unset):
            raise ValueError(f"task_type must match const 'concatenate', got '{task_type}'")

        processing_mode = cast(Literal["transform"] | Unset, d.pop("processing_mode", UNSET))
        if processing_mode != "transform" and not isinstance(processing_mode, Unset):
            raise ValueError(f"processing_mode must match const 'transform', got '{processing_mode}'")

        concatenate_request = cls(
            query=query,
            input_artifacts=input_artifacts,
            context_artifacts=context_artifacts,
            label=label,
            description=description,
            task_id=task_id,
            replaces_task_id=replaces_task_id,
            twin_artifact_id=twin_artifact_id,
            task_type=task_type,
            processing_mode=processing_mode,
        )

        concatenate_request.additional_properties = d
        return concatenate_request

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
