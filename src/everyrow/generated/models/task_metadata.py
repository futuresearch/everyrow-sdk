from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, cast
from uuid import UUID

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.task_metadata_cols_to_rename_type_0 import TaskMetadataColsToRenameType0


T = TypeVar("T", bound="TaskMetadata")


@_attrs_define
class TaskMetadata:
    """
    Attributes:
        llm (None | str | Unset):
        n_subtask_total (int | None | Unset):
        n_subtask_successes (int | None | Unset):
        n_subtask_failures (int | None | Unset):
        trace_url (None | str | Unset):
        otel_trace_url (None | str | Unset):
        langfuse_trace_url (None | str | Unset):
        duplicate_from_artifact_id (None | Unset | UUID):
        duplicate_from_task_id (None | Unset | UUID):
        cols_to_rename (None | TaskMetadataColsToRenameType0 | Unset):
        cols_to_drop (list[Any] | None | Unset):
    """

    llm: None | str | Unset = UNSET
    n_subtask_total: int | None | Unset = UNSET
    n_subtask_successes: int | None | Unset = UNSET
    n_subtask_failures: int | None | Unset = UNSET
    trace_url: None | str | Unset = UNSET
    otel_trace_url: None | str | Unset = UNSET
    langfuse_trace_url: None | str | Unset = UNSET
    duplicate_from_artifact_id: None | Unset | UUID = UNSET
    duplicate_from_task_id: None | Unset | UUID = UNSET
    cols_to_rename: None | TaskMetadataColsToRenameType0 | Unset = UNSET
    cols_to_drop: list[Any] | None | Unset = UNSET
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        from ..models.task_metadata_cols_to_rename_type_0 import TaskMetadataColsToRenameType0

        llm: None | str | Unset
        if isinstance(self.llm, Unset):
            llm = UNSET
        else:
            llm = self.llm

        n_subtask_total: int | None | Unset
        if isinstance(self.n_subtask_total, Unset):
            n_subtask_total = UNSET
        else:
            n_subtask_total = self.n_subtask_total

        n_subtask_successes: int | None | Unset
        if isinstance(self.n_subtask_successes, Unset):
            n_subtask_successes = UNSET
        else:
            n_subtask_successes = self.n_subtask_successes

        n_subtask_failures: int | None | Unset
        if isinstance(self.n_subtask_failures, Unset):
            n_subtask_failures = UNSET
        else:
            n_subtask_failures = self.n_subtask_failures

        trace_url: None | str | Unset
        if isinstance(self.trace_url, Unset):
            trace_url = UNSET
        else:
            trace_url = self.trace_url

        otel_trace_url: None | str | Unset
        if isinstance(self.otel_trace_url, Unset):
            otel_trace_url = UNSET
        else:
            otel_trace_url = self.otel_trace_url

        langfuse_trace_url: None | str | Unset
        if isinstance(self.langfuse_trace_url, Unset):
            langfuse_trace_url = UNSET
        else:
            langfuse_trace_url = self.langfuse_trace_url

        duplicate_from_artifact_id: None | str | Unset
        if isinstance(self.duplicate_from_artifact_id, Unset):
            duplicate_from_artifact_id = UNSET
        elif isinstance(self.duplicate_from_artifact_id, UUID):
            duplicate_from_artifact_id = str(self.duplicate_from_artifact_id)
        else:
            duplicate_from_artifact_id = self.duplicate_from_artifact_id

        duplicate_from_task_id: None | str | Unset
        if isinstance(self.duplicate_from_task_id, Unset):
            duplicate_from_task_id = UNSET
        elif isinstance(self.duplicate_from_task_id, UUID):
            duplicate_from_task_id = str(self.duplicate_from_task_id)
        else:
            duplicate_from_task_id = self.duplicate_from_task_id

        cols_to_rename: dict[str, Any] | None | Unset
        if isinstance(self.cols_to_rename, Unset):
            cols_to_rename = UNSET
        elif isinstance(self.cols_to_rename, TaskMetadataColsToRenameType0):
            cols_to_rename = self.cols_to_rename.to_dict()
        else:
            cols_to_rename = self.cols_to_rename

        cols_to_drop: list[Any] | None | Unset
        if isinstance(self.cols_to_drop, Unset):
            cols_to_drop = UNSET
        elif isinstance(self.cols_to_drop, list):
            cols_to_drop = self.cols_to_drop

        else:
            cols_to_drop = self.cols_to_drop

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if llm is not UNSET:
            field_dict["llm"] = llm
        if n_subtask_total is not UNSET:
            field_dict["n_subtask_total"] = n_subtask_total
        if n_subtask_successes is not UNSET:
            field_dict["n_subtask_successes"] = n_subtask_successes
        if n_subtask_failures is not UNSET:
            field_dict["n_subtask_failures"] = n_subtask_failures
        if trace_url is not UNSET:
            field_dict["trace_url"] = trace_url
        if otel_trace_url is not UNSET:
            field_dict["otel_trace_url"] = otel_trace_url
        if langfuse_trace_url is not UNSET:
            field_dict["langfuse_trace_url"] = langfuse_trace_url
        if duplicate_from_artifact_id is not UNSET:
            field_dict["duplicate_from_artifact_id"] = duplicate_from_artifact_id
        if duplicate_from_task_id is not UNSET:
            field_dict["duplicate_from_task_id"] = duplicate_from_task_id
        if cols_to_rename is not UNSET:
            field_dict["cols_to_rename"] = cols_to_rename
        if cols_to_drop is not UNSET:
            field_dict["cols_to_drop"] = cols_to_drop

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.task_metadata_cols_to_rename_type_0 import TaskMetadataColsToRenameType0

        d = dict(src_dict)

        def _parse_llm(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        llm = _parse_llm(d.pop("llm", UNSET))

        def _parse_n_subtask_total(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        n_subtask_total = _parse_n_subtask_total(d.pop("n_subtask_total", UNSET))

        def _parse_n_subtask_successes(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        n_subtask_successes = _parse_n_subtask_successes(d.pop("n_subtask_successes", UNSET))

        def _parse_n_subtask_failures(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        n_subtask_failures = _parse_n_subtask_failures(d.pop("n_subtask_failures", UNSET))

        def _parse_trace_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        trace_url = _parse_trace_url(d.pop("trace_url", UNSET))

        def _parse_otel_trace_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        otel_trace_url = _parse_otel_trace_url(d.pop("otel_trace_url", UNSET))

        def _parse_langfuse_trace_url(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        langfuse_trace_url = _parse_langfuse_trace_url(d.pop("langfuse_trace_url", UNSET))

        def _parse_duplicate_from_artifact_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                duplicate_from_artifact_id_type_0 = UUID(data)

                return duplicate_from_artifact_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        duplicate_from_artifact_id = _parse_duplicate_from_artifact_id(d.pop("duplicate_from_artifact_id", UNSET))

        def _parse_duplicate_from_task_id(data: object) -> None | Unset | UUID:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                duplicate_from_task_id_type_0 = UUID(data)

                return duplicate_from_task_id_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | Unset | UUID, data)

        duplicate_from_task_id = _parse_duplicate_from_task_id(d.pop("duplicate_from_task_id", UNSET))

        def _parse_cols_to_rename(data: object) -> None | TaskMetadataColsToRenameType0 | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                cols_to_rename_type_0 = TaskMetadataColsToRenameType0.from_dict(data)

                return cols_to_rename_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(None | TaskMetadataColsToRenameType0 | Unset, data)

        cols_to_rename = _parse_cols_to_rename(d.pop("cols_to_rename", UNSET))

        def _parse_cols_to_drop(data: object) -> list[Any] | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                cols_to_drop_type_0 = cast(list[Any], data)

                return cols_to_drop_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(list[Any] | None | Unset, data)

        cols_to_drop = _parse_cols_to_drop(d.pop("cols_to_drop", UNSET))

        task_metadata = cls(
            llm=llm,
            n_subtask_total=n_subtask_total,
            n_subtask_successes=n_subtask_successes,
            n_subtask_failures=n_subtask_failures,
            trace_url=trace_url,
            otel_trace_url=otel_trace_url,
            langfuse_trace_url=langfuse_trace_url,
            duplicate_from_artifact_id=duplicate_from_artifact_id,
            duplicate_from_task_id=duplicate_from_task_id,
            cols_to_rename=cols_to_rename,
            cols_to_drop=cols_to_drop,
        )

        task_metadata.additional_properties = d
        return task_metadata

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
