from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="DeepRankPublicParams")


@_attrs_define
class DeepRankPublicParams:
    """
    Attributes:
        task (str): The task for each agent to perform.
        response_schema (Any): The schema for each agent's response.
        field_to_sort_by (str): The field to use when sorting the output artifacts. Must be a top-level field in the
            root model of response_schema. Typically has a numeric type but not necessarily.
        ascending_order (bool | Unset): If true, sort the output artifacts in ascending order according to
            field_to_sort_by Default: True.
        include_provenance_and_notes (bool | Unset): Whether to include an additional provenance and notes field in the
            output and prompt Default: True.
        preview (bool | Unset): When true, process only the first few inputs Default: False.
    """

    task: str
    response_schema: Any
    field_to_sort_by: str
    ascending_order: bool | Unset = True
    include_provenance_and_notes: bool | Unset = True
    preview: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task = self.task

        response_schema = self.response_schema

        field_to_sort_by = self.field_to_sort_by

        ascending_order = self.ascending_order

        include_provenance_and_notes = self.include_provenance_and_notes

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task": task,
                "response_schema": response_schema,
                "field_to_sort_by": field_to_sort_by,
            }
        )
        if ascending_order is not UNSET:
            field_dict["ascending_order"] = ascending_order
        if include_provenance_and_notes is not UNSET:
            field_dict["include_provenance_and_notes"] = include_provenance_and_notes
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task = d.pop("task")

        response_schema = d.pop("response_schema")

        field_to_sort_by = d.pop("field_to_sort_by")

        ascending_order = d.pop("ascending_order", UNSET)

        include_provenance_and_notes = d.pop("include_provenance_and_notes", UNSET)

        preview = d.pop("preview", UNSET)

        deep_rank_public_params = cls(
            task=task,
            response_schema=response_schema,
            field_to_sort_by=field_to_sort_by,
            ascending_order=ascending_order,
            include_provenance_and_notes=include_provenance_and_notes,
            preview=preview,
        )

        deep_rank_public_params.additional_properties = d
        return deep_rank_public_params

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
