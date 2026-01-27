from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.response_schema_type import ResponseSchemaType
from ..types import UNSET, Unset

T = TypeVar("T", bound="DeepScreenPublicParams")


@_attrs_define
class DeepScreenPublicParams:
    """
    Attributes:
        task (str): The task instructions for each agent to perform.
        batch_size (int | None | Unset):  Default: 10.
        response_schema (Any | Unset):  Default: {'_model_name': 'Root', 'answer': {'description': 'The response
            answer', 'optional': False, 'type': 'str'}}.
        response_schema_type (ResponseSchemaType | Unset): Type of response schema format.
        include_provenance_and_notes (bool | Unset): Whether to include an additional provenance and notes field in the
            output and prompt Default: True.
        preview (bool | Unset): When true, process only the first few inputs Default: False.
    """

    task: str
    batch_size: int | None | Unset = 10
    response_schema: Any | Unset = {
        "_model_name": "Root",
        "answer": {"description": "The response answer", "optional": False, "type": "str"},
    }
    response_schema_type: ResponseSchemaType | Unset = UNSET
    include_provenance_and_notes: bool | Unset = True
    preview: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task = self.task

        batch_size: int | None | Unset
        if isinstance(self.batch_size, Unset):
            batch_size = UNSET
        else:
            batch_size = self.batch_size

        response_schema = self.response_schema

        response_schema_type: str | Unset = UNSET
        if not isinstance(self.response_schema_type, Unset):
            response_schema_type = self.response_schema_type.value

        include_provenance_and_notes = self.include_provenance_and_notes

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task": task,
            }
        )
        if batch_size is not UNSET:
            field_dict["batch_size"] = batch_size
        if response_schema is not UNSET:
            field_dict["response_schema"] = response_schema
        if response_schema_type is not UNSET:
            field_dict["response_schema_type"] = response_schema_type
        if include_provenance_and_notes is not UNSET:
            field_dict["include_provenance_and_notes"] = include_provenance_and_notes
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task = d.pop("task")

        def _parse_batch_size(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        batch_size = _parse_batch_size(d.pop("batch_size", UNSET))

        response_schema = d.pop("response_schema", UNSET)

        _response_schema_type = d.pop("response_schema_type", UNSET)
        response_schema_type: ResponseSchemaType | Unset
        if isinstance(_response_schema_type, Unset):
            response_schema_type = UNSET
        else:
            response_schema_type = ResponseSchemaType(_response_schema_type)

        include_provenance_and_notes = d.pop("include_provenance_and_notes", UNSET)

        preview = d.pop("preview", UNSET)

        deep_screen_public_params = cls(
            task=task,
            batch_size=batch_size,
            response_schema=response_schema,
            response_schema_type=response_schema_type,
            include_provenance_and_notes=include_provenance_and_notes,
            preview=preview,
        )

        deep_screen_public_params.additional_properties = d
        return deep_screen_public_params

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
