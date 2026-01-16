from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.data_frame_method import DataFrameMethod

T = TypeVar("T", bound="FilterQueryParams")


@_attrs_define
class FilterQueryParams:
    """
    Attributes:
        field_name (str):
        dataframe_method (DataFrameMethod):
        rhs (bool | float | int | str):
        invert_mask (bool):
    """

    field_name: str
    dataframe_method: DataFrameMethod
    rhs: bool | float | int | str
    invert_mask: bool
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        field_name = self.field_name

        dataframe_method = self.dataframe_method.value

        rhs: bool | float | int | str
        rhs = self.rhs

        invert_mask = self.invert_mask

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "field_name": field_name,
                "dataframe_method": dataframe_method,
                "rhs": rhs,
                "invert_mask": invert_mask,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        field_name = d.pop("field_name")

        dataframe_method = DataFrameMethod(d.pop("dataframe_method"))

        def _parse_rhs(data: object) -> bool | float | int | str:
            return cast(bool | float | int | str, data)

        rhs = _parse_rhs(d.pop("rhs"))

        invert_mask = d.pop("invert_mask")

        filter_query_params = cls(
            field_name=field_name,
            dataframe_method=dataframe_method,
            rhs=rhs,
            invert_mask=invert_mask,
        )

        filter_query_params.additional_properties = d
        return filter_query_params

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
