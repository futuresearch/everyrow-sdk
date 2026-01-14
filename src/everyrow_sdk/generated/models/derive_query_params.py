from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from attrs import define as _attrs_define
from attrs import field as _attrs_field

if TYPE_CHECKING:
    from ..models.derive_expression import DeriveExpression


T = TypeVar("T", bound="DeriveQueryParams")


@_attrs_define
class DeriveQueryParams:
    """
    Attributes:
        expressions (list[DeriveExpression]): List of column derivations to apply
    """

    expressions: list[DeriveExpression]
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        expressions = []
        for expressions_item_data in self.expressions:
            expressions_item = expressions_item_data.to_dict()
            expressions.append(expressions_item)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "expressions": expressions,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.derive_expression import DeriveExpression

        d = dict(src_dict)
        expressions = []
        _expressions = d.pop("expressions")
        for expressions_item_data in _expressions:
            expressions_item = DeriveExpression.from_dict(expressions_item_data)

            expressions.append(expressions_item)

        derive_query_params = cls(
            expressions=expressions,
        )

        derive_query_params.additional_properties = d
        return derive_query_params

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
