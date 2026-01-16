from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.llm_enum import LLMEnum
from ..models.submit_chat_task_body_selected_task_type_type_0 import SubmitChatTaskBodySelectedTaskTypeType0
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
    from ..models.reduce_agent_request_params import ReduceAgentRequestParams
    from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
    from ..models.upload_csv_payload import UploadCsvPayload


T = TypeVar("T", bound="SubmitChatTaskBody")


@_attrs_define
class SubmitChatTaskBody:
    """
    Attributes:
        query (str):
        session_id (UUID):
        conversation_id (UUID):
        model (LLMEnum | Unset):
        partial_payload (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest | DropColumnsRequest | FilterRequest |
            FlattenRequest | GroupByRequest | JoinRequest | MapAgentRequestParams | MapMultiAgentRequestParams | None |
            ReduceAgentRequestParams | ReduceMultiAgentRequestParams | Unset | UploadCsvPayload): Partially filled in task
            config values
        hide_on_frontend (bool | Unset): When true, hide the submitted user message from frontend displays. Default:
            False.
        enable_preview_loop (bool | Unset): Enable preview loop for agent tasks Default: False.
        max_feedback_rounds (int | None | Unset): Maximum number of feedback rounds for interagent tasks
        selected_task_type (None | SubmitChatTaskBodySelectedTaskTypeType0 | Unset): When set, prepend task type
            instruction to user query for Autocohort
    """

    query: str
    session_id: UUID
    conversation_id: UUID
    model: LLMEnum | Unset = UNSET
    partial_payload: (
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
        | None
        | ReduceAgentRequestParams
        | ReduceMultiAgentRequestParams
        | Unset
        | UploadCsvPayload
    ) = UNSET
    hide_on_frontend: bool | Unset = False
    enable_preview_loop: bool | Unset = False
    max_feedback_rounds: int | None | Unset = UNSET
    selected_task_type: None | SubmitChatTaskBodySelectedTaskTypeType0 | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
        from ..models.reduce_agent_request_params import ReduceAgentRequestParams
        from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
        from ..models.upload_csv_payload import UploadCsvPayload

        query = self.query

        session_id = str(self.session_id)

        conversation_id = str(self.conversation_id)

        model: str | Unset = UNSET
        if not isinstance(self.model, Unset):
            model = self.model.value

        partial_payload: dict[str, Any] | None | Unset
        if isinstance(self.partial_payload, Unset):
            partial_payload = UNSET
        elif isinstance(self.partial_payload, MapAgentRequestParams):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, ReduceAgentRequestParams):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, FilterRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DeriveRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, JoinRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, ConcatenateRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DropColumnsRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DedupeRequestParams):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, UploadCsvPayload):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, MapMultiAgentRequestParams):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, ReduceMultiAgentRequestParams):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, CreateRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, CreateGroupRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, FlattenRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, GroupByRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DeepRankRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DeepMergeRequest):
            partial_payload = self.partial_payload.to_dict()
        elif isinstance(self.partial_payload, DeepScreenRequest):
            partial_payload = self.partial_payload.to_dict()
        else:
            partial_payload = self.partial_payload

        hide_on_frontend = self.hide_on_frontend

        enable_preview_loop = self.enable_preview_loop

        max_feedback_rounds: int | None | Unset
        if isinstance(self.max_feedback_rounds, Unset):
            max_feedback_rounds = UNSET
        else:
            max_feedback_rounds = self.max_feedback_rounds

        selected_task_type: None | str | Unset
        if isinstance(self.selected_task_type, Unset):
            selected_task_type = UNSET
        elif isinstance(self.selected_task_type, SubmitChatTaskBodySelectedTaskTypeType0):
            selected_task_type = self.selected_task_type.value
        else:
            selected_task_type = self.selected_task_type

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "query": query,
                "session_id": session_id,
                "conversation_id": conversation_id,
            }
        )
        if model is not UNSET:
            field_dict["model"] = model
        if partial_payload is not UNSET:
            field_dict["partial_payload"] = partial_payload
        if hide_on_frontend is not UNSET:
            field_dict["hide_on_frontend"] = hide_on_frontend
        if enable_preview_loop is not UNSET:
            field_dict["enable_preview_loop"] = enable_preview_loop
        if max_feedback_rounds is not UNSET:
            field_dict["max_feedback_rounds"] = max_feedback_rounds
        if selected_task_type is not UNSET:
            field_dict["selected_task_type"] = selected_task_type

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
        from ..models.reduce_agent_request_params import ReduceAgentRequestParams
        from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
        from ..models.upload_csv_payload import UploadCsvPayload

        d = dict(src_dict)
        query = d.pop("query")

        session_id = UUID(d.pop("session_id"))

        conversation_id = UUID(d.pop("conversation_id"))

        _model = d.pop("model", UNSET)
        model: LLMEnum | Unset
        if isinstance(_model, Unset):
            model = UNSET
        else:
            model = LLMEnum(_model)

        def _parse_partial_payload(
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
            | None
            | ReduceAgentRequestParams
            | ReduceMultiAgentRequestParams
            | Unset
            | UploadCsvPayload
        ):
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
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
                partial_payload_type_0_type_1 = FilterRequest.from_dict(data)

                return partial_payload_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_2 = DeriveRequest.from_dict(data)

                return partial_payload_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_3 = JoinRequest.from_dict(data)

                return partial_payload_type_0_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_4 = ConcatenateRequest.from_dict(data)

                return partial_payload_type_0_type_4
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_5 = DropColumnsRequest.from_dict(data)

                return partial_payload_type_0_type_5
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_6 = DedupeRequestParams.from_dict(data)

                return partial_payload_type_0_type_6
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_7 = UploadCsvPayload.from_dict(data)

                return partial_payload_type_0_type_7
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
                partial_payload_type_0_type_9 = CreateRequest.from_dict(data)

                return partial_payload_type_0_type_9
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_10 = CreateGroupRequest.from_dict(data)

                return partial_payload_type_0_type_10
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_11 = FlattenRequest.from_dict(data)

                return partial_payload_type_0_type_11
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_12 = GroupByRequest.from_dict(data)

                return partial_payload_type_0_type_12
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_13 = DeepRankRequest.from_dict(data)

                return partial_payload_type_0_type_13
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_14 = DeepMergeRequest.from_dict(data)

                return partial_payload_type_0_type_14
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                partial_payload_type_0_type_15 = DeepScreenRequest.from_dict(data)

                return partial_payload_type_0_type_15
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(
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
                | None
                | ReduceAgentRequestParams
                | ReduceMultiAgentRequestParams
                | Unset
                | UploadCsvPayload,
                data,
            )

        partial_payload = _parse_partial_payload(d.pop("partial_payload", UNSET))

        hide_on_frontend = d.pop("hide_on_frontend", UNSET)

        enable_preview_loop = d.pop("enable_preview_loop", UNSET)

        def _parse_max_feedback_rounds(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        max_feedback_rounds = _parse_max_feedback_rounds(d.pop("max_feedback_rounds", UNSET))

        def _parse_selected_task_type(data: object) -> None | SubmitChatTaskBodySelectedTaskTypeType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                selected_task_type_type_0 = SubmitChatTaskBodySelectedTaskTypeType0(data)

                return selected_task_type_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SubmitChatTaskBodySelectedTaskTypeType0 | Unset, data)

        selected_task_type = _parse_selected_task_type(d.pop("selected_task_type", UNSET))

        submit_chat_task_body = cls(
            query=query,
            session_id=session_id,
            conversation_id=conversation_id,
            model=model,
            partial_payload=partial_payload,
            hide_on_frontend=hide_on_frontend,
            enable_preview_loop=enable_preview_loop,
            max_feedback_rounds=max_feedback_rounds,
            selected_task_type=selected_task_type,
        )

        submit_chat_task_body.additional_properties = d
        return submit_chat_task_body

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
