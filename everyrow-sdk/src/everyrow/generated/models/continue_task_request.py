from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.create_artifact_request_data_type_0_item import (
        CreateArtifactRequestDataType0Item,
    )
    from ..models.create_artifact_request_data_type_1 import (
        CreateArtifactRequestDataType1,
    )


T = TypeVar("T", bound="CreateArtifactRequest")


@_attrs_define
class CreateArtifactRequest:
    """
    Attributes:
        data (CreateArtifactRequestDataType1 | list[CreateArtifactRequestDataType0Item]): The data to store as an
            artifact. A list of JSON objects creates a table (group artifact). A single JSON object creates a scalar
            (standalone artifact).
        session_id (None | Unset | UUID): Session ID. If not provided, a new session is auto-created.
    """

    data: CreateArtifactRequestDataType1 | list[CreateArtifactRequestDataType0Item]
    session_id: None | Unset | UUID = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] | list[dict[str, Any]]
        if isinstance(self.data, list):
            data = []
            for data_type_0_item_data in self.data:
                data_type_0_item = data_type_0_item_data.to_dict()
                data.append(data_type_0_item)

        else:
            data = self.data.to_dict()

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
                "data": data,
            }
        )
        if session_id is not UNSET:
            field_dict["session_id"] = session_id

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.create_artifact_request_data_type_0_item import (
            CreateArtifactRequestDataType0Item,
        )
        from ..models.create_artifact_request_data_type_1 import (
            CreateArtifactRequestDataType1,
        )

        d = dict(src_dict)

        def _parse_data(
            data: object,
        ) -> CreateArtifactRequestDataType1 | list[CreateArtifactRequestDataType0Item]:
            try:
                if not isinstance(data, list):
                    raise TypeError()
                data_type_0 = []
                _data_type_0 = data
                for data_type_0_item_data in _data_type_0:
                    data_type_0_item = CreateArtifactRequestDataType0Item.from_dict(
                        data_type_0_item_data
                    )

                    data_type_0.append(data_type_0_item)

                return data_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            data_type_1 = CreateArtifactRequestDataType1.from_dict(data)

            return data_type_1

        data = _parse_data(d.pop("data"))

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

        create_artifact_request = cls(
            data=data,
            session_id=session_id,
        )

        create_artifact_request.additional_properties = d
        return create_artifact_request

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
