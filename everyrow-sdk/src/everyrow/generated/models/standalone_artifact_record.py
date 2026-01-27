from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.artifact_status import ArtifactStatus
from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.aux_data import AuxData
    from ..models.standalone_artifact_record_analysis_type_0 import StandaloneArtifactRecordAnalysisType0
    from ..models.standalone_artifact_record_metadata_type_0 import StandaloneArtifactRecordMetadataType0
    from ..models.standalone_artifact_record_trace_mapping_type_0 import StandaloneArtifactRecordTraceMappingType0


T = TypeVar("T", bound="StandaloneArtifactRecord")


@_attrs_define
class StandaloneArtifactRecord:
    """
    Attributes:
        uid (UUID):
        type_ (Literal['standalone']):
        data (Any):
        metadata (None | StandaloneArtifactRecordMetadataType0 | Unset):
        label (None | str | Unset):
        aux_data (AuxData | Unset):
        status (ArtifactStatus | Unset):
        trace_mapping (None | StandaloneArtifactRecordTraceMappingType0 | Unset):
        original_id (None | Unset | UUID):
        index_in_group (int | None | Unset):
        current_iteration (int | None | Unset):
        analysis (None | StandaloneArtifactRecordAnalysisType0 | Unset):
    """

    uid: UUID
    type_: Literal["standalone"]
    data: Any
    metadata: None | StandaloneArtifactRecordMetadataType0 | Unset = UNSET
    label: None | str | Unset = UNSET
    aux_data: AuxData | Unset = UNSET
    status: ArtifactStatus | Unset = UNSET
    trace_mapping: None | StandaloneArtifactRecordTraceMappingType0 | Unset = UNSET
    original_id: None | Unset | UUID = UNSET
    index_in_group: int | None | Unset = UNSET
    current_iteration: int | None | Unset = UNSET
    analysis: None | StandaloneArtifactRecordAnalysisType0 | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.standalone_artifact_record_analysis_type_0 import StandaloneArtifactRecordAnalysisType0
        from ..models.standalone_artifact_record_metadata_type_0 import StandaloneArtifactRecordMetadataType0
        from ..models.standalone_artifact_record_trace_mapping_type_0 import StandaloneArtifactRecordTraceMappingType0

        uid = str(self.uid)

        type_ = self.type_

        data = self.data

        metadata: dict[str, Any] | None | Unset
        if isinstance(self.metadata, Unset):
            metadata = UNSET
        elif isinstance(self.metadata, StandaloneArtifactRecordMetadataType0):
            metadata = self.metadata.to_dict()
        else:
            metadata = self.metadata

        label: None | str | Unset
        if isinstance(self.label, Unset):
            label = UNSET
        else:
            label = self.label

        aux_data: dict[str, Any] | Unset = UNSET
        if not isinstance(self.aux_data, Unset):
            aux_data = self.aux_data.to_dict()

        status: str | Unset = UNSET
        if not isinstance(self.status, Unset):
            status = self.status.value

        trace_mapping: dict[str, Any] | None | Unset
        if isinstance(self.trace_mapping, Unset):
            trace_mapping = UNSET
        elif isinstance(self.trace_mapping, StandaloneArtifactRecordTraceMappingType0):
            trace_mapping = self.trace_mapping.to_dict()
        else:
            trace_mapping = self.trace_mapping

        original_id: None | str | Unset
        if isinstance(self.original_id, Unset):
            original_id = UNSET
        elif isinstance(self.original_id, UUID):
            original_id = str(self.original_id)
        else:
            original_id = self.original_id

        index_in_group: int | None | Unset
        if isinstance(self.index_in_group, Unset):
            index_in_group = UNSET
        else:
            index_in_group = self.index_in_group

        current_iteration: int | None | Unset
        if isinstance(self.current_iteration, Unset):
            current_iteration = UNSET
        else:
            current_iteration = self.current_iteration

        analysis: dict[str, Any] | None | Unset
        if isinstance(self.analysis, Unset):
            analysis = UNSET
        elif isinstance(self.analysis, StandaloneArtifactRecordAnalysisType0):
            analysis = self.analysis.to_dict()
        else:
            analysis = self.analysis

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "uid": uid,
                "type": type_,
                "data": data,
            }
        )
        if metadata is not UNSET:
            field_dict["metadata"] = metadata
        if label is not UNSET:
            field_dict["label"] = label
        if aux_data is not UNSET:
            field_dict["aux_data"] = aux_data
        if status is not UNSET:
            field_dict["status"] = status
        if trace_mapping is not UNSET:
            field_dict["trace_mapping"] = trace_mapping
        if original_id is not UNSET:
            field_dict["original_id"] = original_id
        if index_in_group is not UNSET:
            field_dict["index_in_group"] = index_in_group
        if current_iteration is not UNSET:
            field_dict["current_iteration"] = current_iteration
        if analysis is not UNSET:
            field_dict["analysis"] = analysis

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.aux_data import AuxData
        from ..models.standalone_artifact_record_analysis_type_0 import StandaloneArtifactRecordAnalysisType0
        from ..models.standalone_artifact_record_metadata_type_0 import StandaloneArtifactRecordMetadataType0
        from ..models.standalone_artifact_record_trace_mapping_type_0 import StandaloneArtifactRecordTraceMappingType0

        d = dict(src_dict)
        uid = UUID(d.pop("uid"))

        type_ = cast(Literal["standalone"], d.pop("type"))
        if type_ != "standalone":
            raise ValueError(f"type must match const 'standalone', got '{type_}'")

        data = d.pop("data")

        def _parse_metadata(data: object) -> None | StandaloneArtifactRecordMetadataType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                metadata_type_0 = StandaloneArtifactRecordMetadataType0.from_dict(data)

                return metadata_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | StandaloneArtifactRecordMetadataType0 | Unset, data)

        metadata = _parse_metadata(d.pop("metadata", UNSET))

        def _parse_label(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        label = _parse_label(d.pop("label", UNSET))

        _aux_data = d.pop("aux_data", UNSET)
        aux_data: AuxData | Unset
        if isinstance(_aux_data, Unset):
            aux_data = UNSET
        else:
            aux_data = AuxData.from_dict(_aux_data)

        _status = d.pop("status", UNSET)
        status: ArtifactStatus | Unset
        if isinstance(_status, Unset):
            status = UNSET
        else:
            status = ArtifactStatus(_status)

        def _parse_trace_mapping(data: object) -> None | StandaloneArtifactRecordTraceMappingType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                trace_mapping_type_0 = StandaloneArtifactRecordTraceMappingType0.from_dict(data)

                return trace_mapping_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | StandaloneArtifactRecordTraceMappingType0 | Unset, data)

        trace_mapping = _parse_trace_mapping(d.pop("trace_mapping", UNSET))

        def _parse_original_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                original_id_type_0 = UUID(data)

                return original_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        original_id = _parse_original_id(d.pop("original_id", UNSET))

        def _parse_index_in_group(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        index_in_group = _parse_index_in_group(d.pop("index_in_group", UNSET))

        def _parse_current_iteration(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        current_iteration = _parse_current_iteration(d.pop("current_iteration", UNSET))

        def _parse_analysis(data: object) -> None | StandaloneArtifactRecordAnalysisType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                analysis_type_0 = StandaloneArtifactRecordAnalysisType0.from_dict(data)

                return analysis_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | StandaloneArtifactRecordAnalysisType0 | Unset, data)

        analysis = _parse_analysis(d.pop("analysis", UNSET))

        standalone_artifact_record = cls(
            uid=uid,
            type_=type_,
            data=data,
            metadata=metadata,
            label=label,
            aux_data=aux_data,
            status=status,
            trace_mapping=trace_mapping,
            original_id=original_id,
            index_in_group=index_in_group,
            current_iteration=current_iteration,
            analysis=analysis,
        )

        standalone_artifact_record.additional_properties = d
        return standalone_artifact_record

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
