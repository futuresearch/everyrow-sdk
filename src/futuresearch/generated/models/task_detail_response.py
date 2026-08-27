from __future__ import annotations

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field
from dateutil.parser import isoparse

from ..models.public_task_type import PublicTaskType
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.forecast_spec import ForecastSpec


T = TypeVar("T", bound="TaskDetailResponse")


@_attrs_define
class TaskDetailResponse:
    """The parameters a task was submitted with

    Attributes:
        task_id (UUID): The task ID
        session_id (UUID): The session this task belongs to
        task_type (PublicTaskType):
        created_at (datetime.datetime | None): When the task was created
        label (None | str | Unset): Human-readable task label (LLM-generated)
        spec (ForecastSpec | None | Unset): Operation parameters. Present for forecast tasks; null for the operation
            types that do not yet publish a spec.
    """

    task_id: UUID
    session_id: UUID
    task_type: PublicTaskType
    created_at: datetime.datetime | None
    label: None | str | Unset = UNSET
    spec: ForecastSpec | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.forecast_spec import ForecastSpec

        task_id = str(self.task_id)

        session_id = str(self.session_id)

        task_type = self.task_type.value

        created_at: None | str
        if isinstance(self.created_at, datetime.datetime):
            created_at = self.created_at.isoformat()
        else:
            created_at = self.created_at

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        spec: dict[str, Any] | None | Unset
        if isinstance(self.spec, Unset):
            spec = UNSET
        elif isinstance(self.spec, ForecastSpec):
            spec = self.spec.to_dict()
        else:
            spec = self.spec

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task_id": task_id,
                "session_id": session_id,
                "task_type": task_type,
                "created_at": created_at,
            }
        )
        if label is not UNSET:
            field_dict["label"] = label
        if spec is not UNSET:
            field_dict["spec"] = spec

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.forecast_spec import ForecastSpec

        d = dict(src_dict)
        task_id = UUID(d.pop("task_id"))

        session_id = UUID(d.pop("session_id"))

        task_type = PublicTaskType(d.pop("task_type"))

        def _parse_created_at(data: object) -> datetime.datetime | None:
            if data is None:
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                created_at_type_0 = isoparse(data)

                return created_at_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(datetime.datetime | None, data)

        created_at = _parse_created_at(d.pop("created_at"))

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        def _parse_spec(data: object) -> ForecastSpec | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                spec_type_0 = ForecastSpec.from_dict(data)

                return spec_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(ForecastSpec | None | Unset, data)

        spec = _parse_spec(d.pop("spec", UNSET))

        task_detail_response = cls(
            task_id=task_id,
            session_id=session_id,
            task_type=task_type,
            created_at=created_at,
            label=label,
            spec=spec,
        )

        task_detail_response.additional_properties = d
        return task_detail_response

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
