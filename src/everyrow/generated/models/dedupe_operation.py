from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.dedupe_operation_input_type_1_item import DedupeOperationInputType1Item
    from ..models.dedupe_operation_input_type_2 import DedupeOperationInputType2


T = TypeVar("T", bound="DedupeOperation")


@_attrs_define
class DedupeOperation:
    """
    Attributes:
        input_ (DedupeOperationInputType2 | list[DedupeOperationInputType1Item] | UUID): The input data as a) the ID of
            an existing artifact, b) a single record in the form of a JSON object, or c) a table of records in the form of a
            list of JSON objects
        equivalence_relation (str): Description of what makes two rows equivalent/duplicates
        session_id (None | Unset | UUID): Session ID. If not provided, a new session is auto-created for this task.
    """

    input_: DedupeOperationInputType2 | list[DedupeOperationInputType1Item] | UUID
    equivalence_relation: str
    session_id: None | Unset | UUID = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
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

        equivalence_relation = self.equivalence_relation

        session_id: None | str | Unset
        if isinstance(self.session_id, Unset):
            session_id = UNSET
        elif isinstance(self.session_id, UUID):
            session_id = str(self.session_id)
        else:
            session_id = self.session_id

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "input": input_,
                "equivalence_relation": equivalence_relation,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.dedupe_operation_input_type_1_item import DedupeOperationInputType1Item
        from ..models.dedupe_operation_input_type_2 import DedupeOperationInputType2

        d = dict(src_dict)

        def _parse_input_(data: object) -> DedupeOperationInputType2 | list[DedupeOperationInputType1Item] | UUID:
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
                    input_type_1_item = DedupeOperationInputType1Item.from_dict(input_type_1_item_data)

                    input_type_1.append(input_type_1_item)

                return input_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            input_type_2 = DedupeOperationInputType2.from_dict(data)

            return input_type_2

        input_ = _parse_input_(d.pop("input"))

        equivalence_relation = d.pop("equivalence_relation")

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

        dedupe_operation = cls(
            input_=input_,
            equivalence_relation=equivalence_relation,
            session_id=session_id,
        )

        dedupe_operation.additional_properties = d
        return dedupe_operation

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
