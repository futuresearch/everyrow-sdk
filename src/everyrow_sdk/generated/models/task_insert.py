from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from dateutil.parser import isoparse

from ..models.processing_mode import ProcessingMode
from ..models.task_status import TaskStatus
from ..models.task_type import TaskType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.execution_metadata import ExecutionMetadata
    from ..models.task_insert_query_params import TaskInsertQueryParams
    from ..models.task_metadata import TaskMetadata


T = TypeVar("T", bound="TaskInsert")


@_attrs_define
class TaskInsert:
    """
    Attributes:
        type_ (TaskType):
        processing_method (ProcessingMode):
        query_params (TaskInsertQueryParams):
        session_id (UUID):
        status (TaskStatus):
        id (None | str | Unset | UUID):
        context_artifacts (list[UUID] | None | Unset):
        created_at (datetime.datetime | None | Unset):
        ended_at (datetime.datetime | None | Unset):
        input_artifacts (list[UUID] | None | Unset):
        metadata (None | TaskMetadata | Unset):
        updated_at (datetime.datetime | None | Unset):
        error (None | str | Unset):
        is_group (bool | None | Unset):
        label (None | str | Unset):
        description (None | str | Unset):
        artifact_id (None | Unset | UUID):
        join_with_input (bool | None | Unset):
        execution_metadata (ExecutionMetadata | None | Unset):
        workflow_task_id (None | Unset | UUID):
        workflow_run_id (None | Unset | UUID):
        replaces_task_id (None | Unset | UUID):
        original_task_id (None | Unset | UUID):
        conversation_id (None | Unset | UUID):
        yolo_mode (bool | None | Unset):
        n_preview_iteration (int | None | Unset):
        enable_preview_loop (bool | None | Unset):
        twin_artifact_id (None | Unset | UUID):
    """

    type_: TaskType
    processing_method: ProcessingMode
    query_params: TaskInsertQueryParams
    session_id: UUID
    status: TaskStatus
    id: None | str | Unset | UUID = UNSET
    context_artifacts: list[UUID] | None | Unset = UNSET
    created_at: datetime.datetime | None | Unset = UNSET
    ended_at: datetime.datetime | None | Unset = UNSET
    input_artifacts: list[UUID] | None | Unset = UNSET
    metadata: None | TaskMetadata | Unset = UNSET
    updated_at: datetime.datetime | None | Unset = UNSET
    error: None | str | Unset = UNSET
    is_group: bool | None | Unset = UNSET
    label: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    artifact_id: None | Unset | UUID = UNSET
    join_with_input: bool | None | Unset = UNSET
    execution_metadata: ExecutionMetadata | None | Unset = UNSET
    workflow_task_id: None | Unset | UUID = UNSET
    workflow_run_id: None | Unset | UUID = UNSET
    replaces_task_id: None | Unset | UUID = UNSET
    original_task_id: None | Unset | UUID = UNSET
    conversation_id: None | Unset | UUID = UNSET
    yolo_mode: bool | None | Unset = UNSET
    n_preview_iteration: int | None | Unset = UNSET
    enable_preview_loop: bool | None | Unset = UNSET
    twin_artifact_id: None | Unset | UUID = UNSET

    def to_dict(self) -> dict[str, Any]:
        from ..models.execution_metadata import ExecutionMetadata
        from ..models.task_metadata import TaskMetadata

        type_ = self.type_.value

        processing_method = self.processing_method.value

        query_params = self.query_params.to_dict()

        session_id = str(self.session_id)

        status = self.status.value

        id: None | str | Unset
        if isinstance(self.id, Unset):
            id = UNSET
        elif isinstance(self.id, UUID):
            id = str(self.id)
        else:
            id = self.id

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

        created_at: None | str | Unset
        if isinstance(self.created_at, Unset):
            created_at = UNSET
        elif isinstance(self.created_at, datetime.datetime):
            created_at = self.created_at.isoformat()
        else:
            created_at = self.created_at

        ended_at: None | str | Unset
        if isinstance(self.ended_at, Unset):
            ended_at = UNSET
        elif isinstance(self.ended_at, datetime.datetime):
            ended_at = self.ended_at.isoformat()
        else:
            ended_at = self.ended_at

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

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, TaskMetadata):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        updated_at: None | str | Unset
        if isinstance(self.updated_at, Unset):
            updated_at = UNSET
        elif isinstance(self.updated_at, datetime.datetime):
            updated_at = self.updated_at.isoformat()
        else:
            updated_at = self.updated_at

        error: None | str | Unset
        if isinstance(self.error, Unset):
            error = UNSET
        else:
            error = self.error

        is_group: bool | None | Unset
        if isinstance(self.is_group, Unset):
            is_group = UNSET
        else:
            is_group = self.is_group

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

        artifact_id: None | str | Unset
        if isinstance(self.artifact_id, Unset):
            artifact_id = UNSET
        elif isinstance(self.artifact_id, UUID):
            artifact_id = str(self.artifact_id)
        else:
            artifact_id = self.artifact_id

        join_with_input: bool | None | Unset
        if isinstance(self.join_with_input, Unset):
            join_with_input = UNSET
        else:
            join_with_input = self.join_with_input

        execution_metadata: dict[str, Any] | None | Unset
        if isinstance(self.execution_metadata, Unset):
            execution_metadata = UNSET
        elif isinstance(self.execution_metadata, ExecutionMetadata):
            execution_metadata = self.execution_metadata.to_dict()
        else:
            execution_metadata = self.execution_metadata

        workflow_task_id: None | str | Unset
        if isinstance(self.workflow_task_id, Unset):
            workflow_task_id = UNSET
        elif isinstance(self.workflow_task_id, UUID):
            workflow_task_id = str(self.workflow_task_id)
        else:
            workflow_task_id = self.workflow_task_id

        workflow_run_id: None | str | Unset
        if isinstance(self.workflow_run_id, Unset):
            workflow_run_id = UNSET
        elif isinstance(self.workflow_run_id, UUID):
            workflow_run_id = str(self.workflow_run_id)
        else:
            workflow_run_id = self.workflow_run_id

        replaces_task_id: None | str | Unset
        if isinstance(self.replaces_task_id, Unset):
            replaces_task_id = UNSET
        elif isinstance(self.replaces_task_id, UUID):
            replaces_task_id = str(self.replaces_task_id)
        else:
            replaces_task_id = self.replaces_task_id

        original_task_id: None | str | Unset
        if isinstance(self.original_task_id, Unset):
            original_task_id = UNSET
        elif isinstance(self.original_task_id, UUID):
            original_task_id = str(self.original_task_id)
        else:
            original_task_id = self.original_task_id

        conversation_id: None | str | Unset
        if isinstance(self.conversation_id, Unset):
            conversation_id = UNSET
        elif isinstance(self.conversation_id, UUID):
            conversation_id = str(self.conversation_id)
        else:
            conversation_id = self.conversation_id

        yolo_mode: bool | None | Unset
        if isinstance(self.yolo_mode, Unset):
            yolo_mode = UNSET
        else:
            yolo_mode = self.yolo_mode

        n_preview_iteration: int | None | Unset
        if isinstance(self.n_preview_iteration, Unset):
            n_preview_iteration = UNSET
        else:
            n_preview_iteration = self.n_preview_iteration

        enable_preview_loop: bool | None | Unset
        if isinstance(self.enable_preview_loop, Unset):
            enable_preview_loop = UNSET
        else:
            enable_preview_loop = self.enable_preview_loop

        twin_artifact_id: None | str | Unset
        if isinstance(self.twin_artifact_id, Unset):
            twin_artifact_id = UNSET
        elif isinstance(self.twin_artifact_id, UUID):
            twin_artifact_id = str(self.twin_artifact_id)
        else:
            twin_artifact_id = self.twin_artifact_id

        field_dict: dict[str, Any] = {}

        field_dict.update(
            {
                "type": type_,
                "processing_method": processing_method,
                "query_params": query_params,
                "session_id": session_id,
                "status": status,
            }
        )
        if id is not UNSET:
            field_dict["id"] = id
        if context_artifacts is not UNSET:
            field_dict["context_artifacts"] = context_artifacts
        if created_at is not UNSET:
            field_dict["created_at"] = created_at
        if ended_at is not UNSET:
            field_dict["ended_at"] = ended_at
        if input_artifacts is not UNSET:
            field_dict["input_artifacts"] = input_artifacts
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if updated_at is not UNSET:
            field_dict["updated_at"] = updated_at
        if error is not UNSET:
            field_dict["error"] = error
        if is_group is not UNSET:
            field_dict["is_group"] = is_group
        if label is not UNSET:
            field_dict["label"] = label
        if description is not UNSET:
            field_dict["description"] = description
        if artifact_id is not UNSET:
            field_dict["artifact_id"] = artifact_id
        if join_with_input is not UNSET:
            field_dict["join_with_input"] = join_with_input
        if execution_metadata is not UNSET:
            field_dict["execution_metadata"] = execution_metadata
        if workflow_task_id is not UNSET:
            field_dict["workflow_task_id"] = workflow_task_id
        if workflow_run_id is not UNSET:
            field_dict["workflow_run_id"] = workflow_run_id
        if replaces_task_id is not UNSET:
            field_dict["replaces_task_id"] = replaces_task_id
        if original_task_id is not UNSET:
            field_dict["original_task_id"] = original_task_id
        if conversation_id is not UNSET:
            field_dict["conversation_id"] = conversation_id
        if yolo_mode is not UNSET:
            field_dict["yolo_mode"] = yolo_mode
        if n_preview_iteration is not UNSET:
            field_dict["n_preview_iteration"] = n_preview_iteration
        if enable_preview_loop is not UNSET:
            field_dict["enable_preview_loop"] = enable_preview_loop
        if twin_artifact_id is not UNSET:
            field_dict["twin_artifact_id"] = twin_artifact_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.execution_metadata import ExecutionMetadata
        from ..models.task_insert_query_params import TaskInsertQueryParams
        from ..models.task_metadata import TaskMetadata

        d = dict(src_dict)
        type_ = TaskType(d.pop("type"))

        processing_method = ProcessingMode(d.pop("processing_method"))

        query_params = TaskInsertQueryParams.from_dict(d.pop("query_params"))

        session_id = UUID(d.pop("session_id"))

        status = TaskStatus(d.pop("status"))

        def _parse_id(data: object) -> None | str | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                id_type_0 = UUID(data)

                return id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | str | Unset | UUID, data)

        id = _parse_id(d.pop("id", UNSET))

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

        def _parse_created_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                created_at_type_0 = isoparse(data)

                return created_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        created_at = _parse_created_at(d.pop("created_at", UNSET))

        def _parse_ended_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                ended_at_type_0 = isoparse(data)

                return ended_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        ended_at = _parse_ended_at(d.pop("ended_at", UNSET))

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

        def _parse_metadata(data: object) -> None | TaskMetadata | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = TaskMetadata.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TaskMetadata | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_updated_at(data: object) -> datetime.datetime | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                updated_at_type_0 = isoparse(data)

                return updated_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None | Unset, data)

        updated_at = _parse_updated_at(d.pop("updated_at", UNSET))

        def _parse_error(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        error = _parse_error(d.pop("error", UNSET))

        def _parse_is_group(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        is_group = _parse_is_group(d.pop("is_group", UNSET))

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

        def _parse_artifact_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                artifact_id_type_0 = UUID(data)

                return artifact_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        artifact_id = _parse_artifact_id(d.pop("artifact_id", UNSET))

        def _parse_join_with_input(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        join_with_input = _parse_join_with_input(d.pop("join_with_input", UNSET))

        def _parse_execution_metadata(data: object) -> ExecutionMetadata | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                execution_metadata_type_0 = ExecutionMetadata.from_dict(data)

                return execution_metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ExecutionMetadata | None | Unset, data)

        execution_metadata = _parse_execution_metadata(d.pop("execution_metadata", UNSET))

        def _parse_workflow_task_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                workflow_task_id_type_0 = UUID(data)

                return workflow_task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        workflow_task_id = _parse_workflow_task_id(d.pop("workflow_task_id", UNSET))

        def _parse_workflow_run_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                workflow_run_id_type_0 = UUID(data)

                return workflow_run_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        workflow_run_id = _parse_workflow_run_id(d.pop("workflow_run_id", UNSET))

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

        def _parse_original_task_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                original_task_id_type_0 = UUID(data)

                return original_task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        original_task_id = _parse_original_task_id(d.pop("original_task_id", UNSET))

        def _parse_conversation_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                conversation_id_type_0 = UUID(data)

                return conversation_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        conversation_id = _parse_conversation_id(d.pop("conversation_id", UNSET))

        def _parse_yolo_mode(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        yolo_mode = _parse_yolo_mode(d.pop("yolo_mode", UNSET))

        def _parse_n_preview_iteration(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        n_preview_iteration = _parse_n_preview_iteration(d.pop("n_preview_iteration", UNSET))

        def _parse_enable_preview_loop(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        enable_preview_loop = _parse_enable_preview_loop(d.pop("enable_preview_loop", UNSET))

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

        task_insert = cls(
            type_=type_,
            processing_method=processing_method,
            query_params=query_params,
            session_id=session_id,
            status=status,
            id=id,
            context_artifacts=context_artifacts,
            created_at=created_at,
            ended_at=ended_at,
            input_artifacts=input_artifacts,
            metadata=metadata,
            updated_at=updated_at,
            error=error,
            is_group=is_group,
            label=label,
            description=description,
            artifact_id=artifact_id,
            join_with_input=join_with_input,
            execution_metadata=execution_metadata,
            workflow_task_id=workflow_task_id,
            workflow_run_id=workflow_run_id,
            replaces_task_id=replaces_task_id,
            original_task_id=original_task_id,
            conversation_id=conversation_id,
            yolo_mode=yolo_mode,
            n_preview_iteration=n_preview_iteration,
            enable_preview_loop=enable_preview_loop,
            twin_artifact_id=twin_artifact_id,
        )

        return task_insert
