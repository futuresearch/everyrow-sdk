from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.forecast_spec_forecast_type import ForecastSpecForecastType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.conditional_framing import ConditionalFraming
    from ..models.decision_framing import DecisionFraming
    from ..models.unconditional_framing import UnconditionalFraming


T = TypeVar("T", bound="ForecastSpec")


@_attrs_define
class ForecastSpec:
    """
    Attributes:
        forecast_type (ForecastSpecForecastType): Outcome type: what shape of answer each row gets
        framing (ConditionalFraming | DecisionFraming | UnconditionalFraming): How the outcome is posed
        output_field (None | str | Unset): Name of the forecast quantity (numeric and date outcomes)
        units (None | str | Unset): Units of the forecast quantity (numeric outcomes)
        categories_field (None | str | Unset): Input column whose cell holds the row's categories
        thresholds_field (None | str | Unset): Input column whose cell holds the row's thresholds
    """

    forecast_type: ForecastSpecForecastType
    framing: ConditionalFraming | DecisionFraming | UnconditionalFraming
    output_field: None | str | Unset = UNSET
    units: None | str | Unset = UNSET
    categories_field: None | str | Unset = UNSET
    thresholds_field: None | str | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.conditional_framing import ConditionalFraming
        from ..models.unconditional_framing import UnconditionalFraming

        forecast_type = self.forecast_type.value

        framing: dict[str, Any]
        if isinstance(self.framing, UnconditionalFraming):
            framing = self.framing.to_dict()
        elif isinstance(self.framing, ConditionalFraming):
            framing = self.framing.to_dict()
        else:
            framing = self.framing.to_dict()

        output_field: None | str | Unset
        if isinstance(self.output_field, Unset):
            output_field = UNSET
        else:
            output_field = self.output_field

        units: None | str | Unset
        if isinstance(self.units, Unset):
            units = UNSET
        else:
            units = self.units

        categories_field: None | str | Unset
        if isinstance(self.categories_field, Unset):
            categories_field = UNSET
        else:
            categories_field = self.categories_field

        thresholds_field: None | str | Unset
        if isinstance(self.thresholds_field, Unset):
            thresholds_field = UNSET
        else:
            thresholds_field = self.thresholds_field

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "forecast_type": forecast_type,
                "framing": framing,
            }
        )
        if output_field is not UNSET:
            field_dict["output_field"] = output_field
        if units is not UNSET:
            field_dict["units"] = units
        if categories_field is not UNSET:
            field_dict["categories_field"] = categories_field
        if thresholds_field is not UNSET:
            field_dict["thresholds_field"] = thresholds_field

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.conditional_framing import ConditionalFraming
        from ..models.decision_framing import DecisionFraming
        from ..models.unconditional_framing import UnconditionalFraming

        d = dict(src_dict)
        forecast_type = ForecastSpecForecastType(d.pop("forecast_type"))

        def _parse_framing(data: object) -> ConditionalFraming | DecisionFraming | UnconditionalFraming:
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                framing_type_0 = UnconditionalFraming.from_dict(data)

                return framing_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                framing_type_1 = ConditionalFraming.from_dict(data)

                return framing_type_1
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            if not isinstance(data, dict):
                raise TypeError()
            framing_type_2 = DecisionFraming.from_dict(data)

            return framing_type_2

        framing = _parse_framing(d.pop("framing"))

        def _parse_output_field(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        output_field = _parse_output_field(d.pop("output_field", UNSET))

        def _parse_units(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        units = _parse_units(d.pop("units", UNSET))

        def _parse_categories_field(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        categories_field = _parse_categories_field(d.pop("categories_field", UNSET))

        def _parse_thresholds_field(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        thresholds_field = _parse_thresholds_field(d.pop("thresholds_field", UNSET))

        forecast_spec = cls(
            forecast_type=forecast_type,
            framing=framing,
            output_field=output_field,
            units=units,
            categories_field=categories_field,
            thresholds_field=thresholds_field,
        )

        forecast_spec.additional_properties = d
        return forecast_spec

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
