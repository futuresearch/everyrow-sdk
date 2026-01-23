from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.public_llm import PublicLLM
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.single_agent_operation_input_type_1_item import SingleAgentOperationInputType1Item
    from ..models.single_agent_operation_input_type_2 import SingleAgentOperationInputType2
    from ..models.single_agent_operation_response_schema_type_0 import SingleAgentOperationResponseSchemaType0


T = TypeVar("T", bound="SingleAgentOperation")


@_attrs_define
class SingleAgentOperation:
    """
    Attributes:
        input_ (list[SingleAgentOperationInputType1Item] | SingleAgentOperationInputType2 | UUID): The input data as a)
            the ID of an existing artifact, b) a single record in the form of a JSON object, or c) a table of records in the
            form of a list of JSON objects
        task (str): Instructions for the AI agent
        session_id (None | Unset | UUID): Session ID. If not provided, a new session is auto-created for this task.
        response_schema (None | SingleAgentOperationResponseSchemaType0 | Unset): JSON Schema for the response format.
            If not provided, use default answer schema.
        llm (PublicLLM | Unset):
    """

    input_: list[SingleAgentOperationInputType1Item] | SingleAgentOperationInputType2 | UUID
    task: str
    session_id: None | Unset | UUID = UNSET
    response_schema: None | SingleAgentOperationResponseSchemaType0 | Unset = UNSET
    llm: PublicLLM | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.single_agent_operation_response_schema_type_0 import SingleAgentOperationResponseSchemaType0

        input_: dict[str, Any] | list[dict[str, Any]] | str
        if isinstance(self.input_, UUID):
            input_ = str(self.input_)
        elif isinstance(self.input_, list):
            input_ = []
            for input_type_1_item_data in self.input_:
                input_type_1_item = input_type_1_item_data.to_dict()
                input_.append(input_type_1_item)

        else:
            input_ = self.input_.to_dict()

        task = self.task

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        elif isinstance(self.session_id, UUID):
            session_id = str(self.session_id)
        else:
            session_id = self.session_id

        response_schema: dict[str, Any] | None | Unset
        if isinstance(self.response_schema, Unset):
            response_schema = UNSET
        elif isinstance(self.response_schema, SingleAgentOperationResponseSchemaType0):
            response_schema = self.response_schema.to_dict()
        else:
            response_schema = self.response_schema

        llm: str | Unset = UNSET
        if not isinstance(self.llm, Unset):
            llm = self.llm.value

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
                "task": task,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id
        if response_schema is not UNSET:
            field_dict["response_schema"] = response_schema
        if llm is not UNSET:
            field_dict["llm"] = llm

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.single_agent_operation_input_type_1_item import SingleAgentOperationInputType1Item
        from ..models.single_agent_operation_input_type_2 import SingleAgentOperationInputType2
        from ..models.single_agent_operation_response_schema_type_0 import SingleAgentOperationResponseSchemaType0

        d = dict(src_dict)

        def _parse_input_(
            data: object,
        ) -> list[SingleAgentOperationInputType1Item] | SingleAgentOperationInputType2 | UUID:
            try:
                if not isinstance(data, str):
                    raise TypeError()
                input_type_0 = UUID(data)

                return input_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, list):
                    raise TypeError()
                input_type_1 = []
                _input_type_1 = data
                for input_type_1_item_data in _input_type_1:
                    input_type_1_item = SingleAgentOperationInputType1Item.from_dict(input_type_1_item_data)

                    input_type_1.append(input_type_1_item)

                return input_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            input_type_2 = SingleAgentOperationInputType2.from_dict(data)

            return input_type_2

        input_ = _parse_input_(d.pop("input"))

        task = d.pop("task")

        def _parse_session_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                session_id_type_0 = UUID(data)

                return session_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        session_id = _parse_session_id(d.pop("session_id", UNSET))

        def _parse_response_schema(data: object) -> None | SingleAgentOperationResponseSchemaType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                response_schema_type_0 = SingleAgentOperationResponseSchemaType0.from_dict(data)

                return response_schema_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | SingleAgentOperationResponseSchemaType0 | Unset, data)

        response_schema = _parse_response_schema(d.pop("response_schema", UNSET))

        _llm = d.pop("llm", UNSET)
        llm: PublicLLM | Unset
        if isinstance(_llm, Unset):
            llm = UNSET
        else:
            llm = PublicLLM(_llm)

        single_agent_operation = cls(
            input_=input_,
            task=task,
            session_id=session_id,
            response_schema=response_schema,
            llm=llm,
        )

        single_agent_operation.additional_properties = d
        return single_agent_operation

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
