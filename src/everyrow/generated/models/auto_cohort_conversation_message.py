from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.chat_message_metadata import ChatMessageMetadata
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
    from ..models.tool_response_message import ToolResponseMessage
    from ..models.upload_csv_payload import UploadCsvPayload


T = TypeVar("T", bound="AutoCohortConversationMessage")


@_attrs_define
class AutoCohortConversationMessage:
    """
    Attributes:
        message (MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage):
        metadata (ChatMessageMetadata | Unset):
        parent_message_id (None | Unset | UUID):
        payload_is_suggestion (bool | None | Unset):
        validated_payload (ConcatenateRequest | CreateGroupRequest | CreateRequest | DedupeRequestParams |
            DeepMergeRequest | DeepRankRequest | DeepScreenRequest | DeriveRequest | DropColumnsRequest | FilterRequest |
            FlattenRequest | GroupByRequest | JoinRequest | MapAgentRequestParams | MapMultiAgentRequestParams | None |
            ReduceAgentRequestParams | ReduceMultiAgentRequestParams | Unset | UploadCsvPayload):
        hide_on_frontend (bool | None | Unset):
    """

    message: MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage
    metadata: ChatMessageMetadata | Unset = UNSET
    parent_message_id: None | Unset | UUID = UNSET
    payload_is_suggestion: bool | None | Unset = UNSET
    validated_payload: (
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
    hide_on_frontend: bool | None | Unset = UNSET
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
        from ..models.multi_modal_chat_message import MultiModalChatMessage
        from ..models.reduce_agent_request_params import ReduceAgentRequestParams
        from ..models.reduce_multi_agent_request_params import ReduceMultiAgentRequestParams
        from ..models.simple_chat_message import SimpleChatMessage
        from ..models.simple_chat_message_with_tool_calls import SimpleChatMessageWithToolCalls
        from ..models.upload_csv_payload import UploadCsvPayload

        message: dict[str, Any]
        if isinstance(self.message, SimpleChatMessage):
            message = self.message.to_dict()
        elif isinstance(self.message, MultiModalChatMessage):
            message = self.message.to_dict()
        elif isinstance(self.message, SimpleChatMessageWithToolCalls):
            message = self.message.to_dict()
        else:
            message = self.message.to_dict()

        metadata: dict[str, Any] | Unset = UNSET
        if not isinstance(self.metadata, Unset):
            metadata = self.metadata.to_dict()

        parent_message_id: None | str | Unset
        if isinstance(self.parent_message_id, Unset):
            parent_message_id = UNSET
        elif isinstance(self.parent_message_id, UUID):
            parent_message_id = str(self.parent_message_id)
        else:
            parent_message_id = self.parent_message_id

        payload_is_suggestion: bool | None | Unset
        if isinstance(self.payload_is_suggestion, Unset):
            payload_is_suggestion = UNSET
        else:
            payload_is_suggestion = self.payload_is_suggestion

        validated_payload: dict[str, Any] | None | Unset
        if isinstance(self.validated_payload, Unset):
            validated_payload = UNSET
        elif isinstance(self.validated_payload, MapAgentRequestParams):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, ReduceAgentRequestParams):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, FilterRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DeriveRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, JoinRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, ConcatenateRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DropColumnsRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DedupeRequestParams):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, UploadCsvPayload):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, MapMultiAgentRequestParams):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, ReduceMultiAgentRequestParams):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, CreateRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, CreateGroupRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, FlattenRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, GroupByRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DeepRankRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DeepMergeRequest):
            validated_payload = self.validated_payload.to_dict()
        elif isinstance(self.validated_payload, DeepScreenRequest):
            validated_payload = self.validated_payload.to_dict()
        else:
            validated_payload = self.validated_payload

        hide_on_frontend: bool | None | Unset
        if isinstance(self.hide_on_frontend, Unset):
            hide_on_frontend = UNSET
        else:
            hide_on_frontend = self.hide_on_frontend

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "message": message,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if parent_message_id is not UNSET:
            field_dict["parent_message_id"] = parent_message_id
        if payload_is_suggestion is not UNSET:
            field_dict["payload_is_suggestion"] = payload_is_suggestion
        if validated_payload is not UNSET:
            field_dict["validated_payload"] = validated_payload
        if hide_on_frontend is not UNSET:
            field_dict["hide_on_frontend"] = hide_on_frontend

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.chat_message_metadata import ChatMessageMetadata
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
        from ..models.tool_response_message import ToolResponseMessage
        from ..models.upload_csv_payload import UploadCsvPayload

        d = dict(src_dict)

        def _parse_message(
            data: object,
        ) -> MultiModalChatMessage | SimpleChatMessage | SimpleChatMessageWithToolCalls | ToolResponseMessage:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                message_type_0 = SimpleChatMessage.from_dict(data)

                return message_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                message_type_1 = MultiModalChatMessage.from_dict(data)

                return message_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                message_type_2 = SimpleChatMessageWithToolCalls.from_dict(data)

                return message_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            message_type_3 = ToolResponseMessage.from_dict(data)

            return message_type_3

        message = _parse_message(d.pop("message"))

        _metadata = d.pop("metadata", UNSET)
        metadata: ChatMessageMetadata | Unset
        if isinstance(_metadata, Unset):
            metadata = UNSET
        else:
            metadata = ChatMessageMetadata.from_dict(_metadata)

        def _parse_parent_message_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                parent_message_id_type_0 = UUID(data)

                return parent_message_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        parent_message_id = _parse_parent_message_id(d.pop("parent_message_id", UNSET))

        def _parse_payload_is_suggestion(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        payload_is_suggestion = _parse_payload_is_suggestion(d.pop("payload_is_suggestion", UNSET))

        def _parse_validated_payload(
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
                validated_payload_type_0_type_1 = FilterRequest.from_dict(data)

                return validated_payload_type_0_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_2 = DeriveRequest.from_dict(data)

                return validated_payload_type_0_type_2
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_3 = JoinRequest.from_dict(data)

                return validated_payload_type_0_type_3
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_4 = ConcatenateRequest.from_dict(data)

                return validated_payload_type_0_type_4
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_5 = DropColumnsRequest.from_dict(data)

                return validated_payload_type_0_type_5
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_6 = DedupeRequestParams.from_dict(data)

                return validated_payload_type_0_type_6
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_7 = UploadCsvPayload.from_dict(data)

                return validated_payload_type_0_type_7
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
                validated_payload_type_0_type_9 = CreateRequest.from_dict(data)

                return validated_payload_type_0_type_9
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_10 = CreateGroupRequest.from_dict(data)

                return validated_payload_type_0_type_10
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_11 = FlattenRequest.from_dict(data)

                return validated_payload_type_0_type_11
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_12 = GroupByRequest.from_dict(data)

                return validated_payload_type_0_type_12
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_13 = DeepRankRequest.from_dict(data)

                return validated_payload_type_0_type_13
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_14 = DeepMergeRequest.from_dict(data)

                return validated_payload_type_0_type_14
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                validated_payload_type_0_type_15 = DeepScreenRequest.from_dict(data)

                return validated_payload_type_0_type_15
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

        validated_payload = _parse_validated_payload(d.pop("validated_payload", UNSET))

        def _parse_hide_on_frontend(data: object) -> bool | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(bool | None | Unset, data)

        hide_on_frontend = _parse_hide_on_frontend(d.pop("hide_on_frontend", UNSET))

        auto_cohort_conversation_message = cls(
            message=message,
            metadata=metadata,
            parent_message_id=parent_message_id,
            payload_is_suggestion=payload_is_suggestion,
            validated_payload=validated_payload,
            hide_on_frontend=hide_on_frontend,
        )

        auto_cohort_conversation_message.additional_properties = d
        return auto_cohort_conversation_message

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
