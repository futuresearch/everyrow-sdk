from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.concatenate_request import ConcatenateRequest
    from ..models.create_group_request import CreateGroupRequest
    from ..models.create_request import CreateRequest
    from ..models.dedupe_request_params import DedupeRequestParams
    from ..models.deep_merge_request import DeepMergeRequest
    from ..models.deep_rank_request import DeepRankRequest
    from ..models.deep_screen_request import DeepScreenRequest
    from ..models.derive_request import DeriveRequest
    from ..models.drop_columns_request import DropColumnsRequest
    from ..models.filter_request import FilterRequest
    from ..models.flatten_request import FlattenRequest
    from ..models.group_by_request import GroupByRequest
    from ..models.join_request import JoinRequest
    from ..models.map_agent_request_params import MapAgentRequestParams
    from ..models.map_multi_agent_request_params import MapMultiAgentRequestParams
    from ..models.multi_modal_chat_message import MultiModalChatMessage
    from ..models.reduce_agent_request_params import ReduceAgentRequestParams
    from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
    from ..models.simple_chat_message import SimpleChatMessage
    from ..models.simple_chat_message_with_tool_calls import SimpleChatMessageWithToolCalls
    from ..models.task_metadata import TaskMetadata
    from ..models.tool_response_message import ToolResponseMessage
    from ..models.upload_csv_payload import UploadCsvPayload


T = TypeVar("T", bound="SubmitTaskBody")


@_attrs_define
class SubmitTaskBody:
    """
    Attributes:
        payload (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams | DeepMergeRequest |
            DeepRankRequest | DeepScreenRequest | DeriveRequest | DropColumnsRequest | FilterRequest | FlattenRequest |
            GroupByRequest | JoinRequest | MapAgentRequestParams | MapMultiAgentRequestParams | ReduceAgentRequestParams |
            ReduceMultiAgentRequestParams | UploadCsvPayload):
        session_id (UUID):
        task_id (UUID | Unset): Cohort task ID
        label (None | str | Unset):
        description (None | str | Unset):
        message_history (list[MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls |
            ToolResponseMessage] | Unset): Additional context for generating task descriptions
        workflow_run_id (None | Unset | UUID):
        workflow_task_id (None | Unset | UUID):
        replaces_task_id (None | Unset | UUID): The ID of the task that this task replaces, e.g. full version of a task
            that replaces a preview version
        original_task_id (None | Unset | UUID): The ID of the original task of this chain. In other words, the task
            you'd reach by following the chain of replaces_task_id.
        conversation_id (None | Unset | UUID): The chat conversation this task is associated with (agent loop)
        yolo_mode (bool | Unset): When true, automatically schedule non-preview map tasks without user confirmation.
            Default: False.
        metadata (None | TaskMetadata | Unset): Optional metadata for the task, such as re-execution information
        n_preview_iteration (int | None | Unset):
        enable_preview_loop (bool | Unset): When true, enable the preview loop for the task Default: False.
        twin_artifact_id (None | Unset | UUID): The ID of a reference artifact, e.g. the right table in Deep Merge
            operation.
    """

    payload: (
        ConcatenateRequest
        | CreateGroupRequest
        | CreateRequest
        | DedupeRequestParams
        | DeepMergeRequest
        | DeepRankRequest
        | DeepScreenRequest
        | DeriveRequest
        | DropColumnsRequest
        | FilterRequest
        | FlattenRequest
        | GroupByRequest
        | JoinRequest
        | MapAgentRequestParams
        | MapMultiAgentRequestParams
        | ReduceAgentRequestParams
        | ReduceMultiAgentRequestParams
        | UploadCsvPayload
    )
    session_id: UUID
    task_id: UUID | Unset = UNSET
    label: None | str | Unset = UNSET
    description: None | str | Unset = UNSET
    message_history: (
        list[MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage] | Unset
    ) = UNSET
    workflow_run_id: None | Unset | UUID = UNSET
    workflow_task_id: None | Unset | UUID = UNSET
    replaces_task_id: None | Unset | UUID = UNSET
    original_task_id: None | Unset | UUID = UNSET
    conversation_id: None | Unset | UUID = UNSET
    yolo_mode: bool | Unset = False
    metadata: None | TaskMetadata | Unset = UNSET
    n_preview_iteration: int | None | Unset = UNSET
    enable_preview_loop: bool | Unset = False
    twin_artifact_id: None | Unset | UUID = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.concatenate_request import ConcatenateRequest
        from ..models.create_group_request import CreateGroupRequest
        from ..models.create_request import CreateRequest
        from ..models.dedupe_request_params import DedupeRequestParams
        from ..models.deep_merge_request import DeepMergeRequest
        from ..models.deep_rank_request import DeepRankRequest
        from ..models.derive_request import DeriveRequest
        from ..models.drop_columns_request import DropColumnsRequest
        from ..models.filter_request import FilterRequest
        from ..models.flatten_request import FlattenRequest
        from ..models.group_by_request import GroupByRequest
        from ..models.join_request import JoinRequest
        from ..models.map_agent_request_params import MapAgentRequestParams
        from ..models.map_multi_agent_request_params import MapMultiAgentRequestParams
        from ..models.multi_modal_chat_message import MultiModalChatMessage
        from ..models.reduce_agent_request_params import ReduceAgentRequestParams
        from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
        from ..models.simple_chat_message import SimpleChatMessage
        from ..models.simple_chat_message_with_tool_calls import SimpleChatMessageWithToolCalls
        from ..models.task_metadata import TaskMetadata
        from ..models.upload_csv_payload import UploadCsvPayload

        payload: dict[str, Any]
        if isinstance(self.payload, MapAgentRequestParams):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, ReduceAgentRequestParams):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, FilterRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, DeriveRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, JoinRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, ConcatenateRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, DropColumnsRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, DedupeRequestParams):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, UploadCsvPayload):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, MapMultiAgentRequestParams):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, ReduceMultiAgentRequestParams):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, CreateRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, CreateGroupRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, FlattenRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, GroupByRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, DeepRankRequest):
            payload = self.payload.to_dict()
        elif isinstance(self.payload, DeepMergeRequest):
            payload = self.payload.to_dict()
        else:
            payload = self.payload.to_dict()

        session_id = str(self.session_id)

        task_id: str | Unset = UNSET
        if not isinstance(self.task_id, Unset):
            task_id = str(self.task_id)

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

        message_history: list[dict[str, Any]] | Unset = UNSET
        if not isinstance(self.message_history, Unset):
            message_history = []
            for message_history_item_data in self.message_history:
                message_history_item: dict[str, Any]
                if isinstance(message_history_item_data, SimpleChatMessage):
                    message_history_item = message_history_item_data.to_dict()
                elif isinstance(message_history_item_data, MultiModalChatMessage):
                    message_history_item = message_history_item_data.to_dict()
                elif isinstance(message_history_item_data, SimpleChatMessageWithToolCalls):
                    message_history_item = message_history_item_data.to_dict()
                else:
                    message_history_item = message_history_item_data.to_dict()

                message_history.append(message_history_item)

        workflow_run_id: None | str | Unset
        if isinstance(self.workflow_run_id, Unset):
            workflow_run_id = UNSET
        elif isinstance(self.workflow_run_id, UUID):
            workflow_run_id = str(self.workflow_run_id)
        else:
            workflow_run_id = self.workflow_run_id

        workflow_task_id: None | str | Unset
        if isinstance(self.workflow_task_id, Unset):
            workflow_task_id = UNSET
        elif isinstance(self.workflow_task_id, UUID):
            workflow_task_id = str(self.workflow_task_id)
        else:
            workflow_task_id = self.workflow_task_id

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

        yolo_mode = self.yolo_mode

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, TaskMetadata):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        n_preview_iteration: int | None | Unset
        if isinstance(self.n_preview_iteration, Unset):
            n_preview_iteration = UNSET
        else:
            n_preview_iteration = self.n_preview_iteration

        enable_preview_loop = self.enable_preview_loop

        twin_artifact_id: None | str | Unset
        if isinstance(self.twin_artifact_id, Unset):
            twin_artifact_id = UNSET
        elif isinstance(self.twin_artifact_id, UUID):
            twin_artifact_id = str(self.twin_artifact_id)
        else:
            twin_artifact_id = self.twin_artifact_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "payload": payload,
                "session_id": session_id,
            }
        )
        if task_id is not UNSET:
            field_dict["task_id"] = task_id
        if label is not UNSET:
            field_dict["label"] = label
        if description is not UNSET:
            field_dict["description"] = description
        if message_history is not UNSET:
            field_dict["message_history"] = message_history
        if workflow_run_id is not UNSET:
            field_dict["workflow_run_id"] = workflow_run_id
        if workflow_task_id is not UNSET:
            field_dict["workflow_task_id"] = workflow_task_id
        if replaces_task_id is not UNSET:
            field_dict["replaces_task_id"] = replaces_task_id
        if original_task_id is not UNSET:
            field_dict["original_task_id"] = original_task_id
        if conversation_id is not UNSET:
            field_dict["conversation_id"] = conversation_id
        if yolo_mode is not UNSET:
            field_dict["yolo_mode"] = yolo_mode
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if n_preview_iteration is not UNSET:
            field_dict["n_preview_iteration"] = n_preview_iteration
        if enable_preview_loop is not UNSET:
            field_dict["enable_preview_loop"] = enable_preview_loop
        if twin_artifact_id is not UNSET:
            field_dict["twin_artifact_id"] = twin_artifact_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.concatenate_request import ConcatenateRequest
        from ..models.create_group_request import CreateGroupRequest
        from ..models.create_request import CreateRequest
        from ..models.dedupe_request_params import DedupeRequestParams
        from ..models.deep_merge_request import DeepMergeRequest
        from ..models.deep_rank_request import DeepRankRequest
        from ..models.deep_screen_request import DeepScreenRequest
        from ..models.derive_request import DeriveRequest
        from ..models.drop_columns_request import DropColumnsRequest
        from ..models.filter_request import FilterRequest
        from ..models.flatten_request import FlattenRequest
        from ..models.group_by_request import GroupByRequest
        from ..models.join_request import JoinRequest
        from ..models.map_agent_request_params import MapAgentRequestParams
        from ..models.map_multi_agent_request_params import MapMultiAgentRequestParams
        from ..models.multi_modal_chat_message import MultiModalChatMessage
        from ..models.reduce_agent_request_params import ReduceAgentRequestParams
        from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
        from ..models.simple_chat_message import SimpleChatMessage
        from ..models.simple_chat_message_with_tool_calls import SimpleChatMessageWithToolCalls
        from ..models.task_metadata import TaskMetadata
        from ..models.tool_response_message import ToolResponseMessage
        from ..models.upload_csv_payload import UploadCsvPayload

        d = dict(src_dict)

        def _parse_payload(
            data: object,
        ) -> (
            ConcatenateRequest
            | CreateGroupRequest
            | CreateRequest
            | DedupeRequestParams
            | DeepMergeRequest
            | DeepRankRequest
            | DeepScreenRequest
            | DeriveRequest
            | DropColumnsRequest
            | FilterRequest
            | FlattenRequest
            | GroupByRequest
            | JoinRequest
            | MapAgentRequestParams
            | MapMultiAgentRequestParams
            | ReduceAgentRequestParams
            | ReduceMultiAgentRequestParams
            | UploadCsvPayload
        ):
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_agent_request_payload_type_0 = MapAgentRequestParams.from_dict(data)

                return componentsschemas_agent_request_payload_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_agent_request_payload_type_1 = ReduceAgentRequestParams.from_dict(data)

                return componentsschemas_agent_request_payload_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_1 = FilterRequest.from_dict(data)

                return payload_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_2 = DeriveRequest.from_dict(data)

                return payload_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_3 = JoinRequest.from_dict(data)

                return payload_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_4 = ConcatenateRequest.from_dict(data)

                return payload_type_4
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_5 = DropColumnsRequest.from_dict(data)

                return payload_type_5
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_6 = DedupeRequestParams.from_dict(data)

                return payload_type_6
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_7 = UploadCsvPayload.from_dict(data)

                return payload_type_7
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_multi_agent_request_payload_type_0 = MapMultiAgentRequestParams.from_dict(data)

                return componentsschemas_multi_agent_request_payload_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                componentsschemas_multi_agent_request_payload_type_1 = ReduceMultiAgentRequestParams.from_dict(data)

                return componentsschemas_multi_agent_request_payload_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_9 = CreateRequest.from_dict(data)

                return payload_type_9
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_10 = CreateGroupRequest.from_dict(data)

                return payload_type_10
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_11 = FlattenRequest.from_dict(data)

                return payload_type_11
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_12 = GroupByRequest.from_dict(data)

                return payload_type_12
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_13 = DeepRankRequest.from_dict(data)

                return payload_type_13
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                payload_type_14 = DeepMergeRequest.from_dict(data)

                return payload_type_14
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            payload_type_15 = DeepScreenRequest.from_dict(data)

            return payload_type_15

        payload = _parse_payload(d.pop("payload"))

        session_id = UUID(d.pop("session_id"))

        _task_id = d.pop("task_id", UNSET)
        task_id: UUID | Unset
        if isinstance(_task_id, Unset):
            task_id = UNSET
        else:
            task_id = UUID(_task_id)

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

        _message_history = d.pop("message_history", UNSET)
        message_history: (
            list[MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage]
            | Unset
        ) = UNSET
        if _message_history is not UNSET:
            message_history = []
            for message_history_item_data in _message_history:

                def _parse_message_history_item(
                    data: object,
                ) -> MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage:
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        message_history_item_type_0 = SimpleChatMessage.from_dict(data)

                        return message_history_item_type_0
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        message_history_item_type_1 = MultiModalChatMessage.from_dict(data)

                        return message_history_item_type_1
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    try:
                        if not isinstance(data, dict):
                            raise TypeError()
                        message_history_item_type_2 = SimpleChatMessageWithToolCalls.from_dict(data)

                        return message_history_item_type_2
                    except (TypeError, ValueError, AttributeError, KeyError):
                        pass
                    if not isinstance(data, dict):
                        raise TypeError()
                    message_history_item_type_3 = ToolResponseMessage.from_dict(data)

                    return message_history_item_type_3

                message_history_item = _parse_message_history_item(message_history_item_data)

                message_history.append(message_history_item)

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

        yolo_mode = d.pop("yolo_mode", UNSET)

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

        def _parse_n_preview_iteration(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        n_preview_iteration = _parse_n_preview_iteration(d.pop("n_preview_iteration", UNSET))

        enable_preview_loop = d.pop("enable_preview_loop", UNSET)

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

        submit_task_body = cls(
            payload=payload,
            session_id=session_id,
            task_id=task_id,
            label=label,
            description=description,
            message_history=message_history,
            workflow_run_id=workflow_run_id,
            workflow_task_id=workflow_task_id,
            replaces_task_id=replaces_task_id,
            original_task_id=original_task_id,
            conversation_id=conversation_id,
            yolo_mode=yolo_mode,
            metadata=metadata,
            n_preview_iteration=n_preview_iteration,
            enable_preview_loop=enable_preview_loop,
            twin_artifact_id=twin_artifact_id,
        )

        submit_task_body.additional_properties = d
        return submit_task_body

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
