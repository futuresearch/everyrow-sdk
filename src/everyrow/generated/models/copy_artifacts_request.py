from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

T = TypeVar("T", bound="CopyArtifactsRequest")


@_attrs_define
class CopyArtifactsRequest:
    """
    Attributes:
        source_artifact_id (UUID):
        target_session_id (UUID):
    """

    source_artifact_id: UUID
    target_session_id: UUID
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        source_artifact_id = str(self.source_artifact_id)

        target_session_id = str(self.target_session_id)

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "source_artifact_id": source_artifact_id,
                "target_session_id": target_session_id,
            }
        )

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        source_artifact_id = UUID(d.pop("source_artifact_id"))

        target_session_id = UUID(d.pop("target_session_id"))

        copy_artifacts_request = cls(
            source_artifact_id=source_artifact_id,
            target_session_id=target_session_id,
        )

        copy_artifacts_request.additional_properties = d
        return copy_artifacts_request

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
