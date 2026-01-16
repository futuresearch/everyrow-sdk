from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.llm_enum import LLMEnum
from ..types import UNSET, Unset

T = TypeVar("T", bound="DeepMergePublicParams")


@_attrs_define
class DeepMergePublicParams:
    """
    Attributes:
        task (str): The task for each agent to perform.
        merge_on_left (None | str | Unset): Column name for merge table
        merge_on_right (None | str | Unset): Column name for merge table
        merge_model (LLMEnum | None | Unset): LLM model for merge operation Default: LLMEnum.GEMINI_3_FLASH_MINIMAL.
        preview (bool | Unset): When true, process only the first few inputs Default: False.
    """

    task: str
    merge_on_left: None | str | Unset = UNSET
    merge_on_right: None | str | Unset = UNSET
    merge_model: LLMEnum | None | Unset = LLMEnum.GEMINI_3_FLASH_MINIMAL
    preview: bool | Unset = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        task = self.task

        merge_on_left: None | str | Unset
        if isinstance(self.merge_on_left, Unset):
            merge_on_left = UNSET
        else:
            merge_on_left = self.merge_on_left

        merge_on_right: None | str | Unset
        if isinstance(self.merge_on_right, Unset):
            merge_on_right = UNSET
        else:
            merge_on_right = self.merge_on_right

        merge_model: None | str | Unset
        if isinstance(self.merge_model, Unset):
            merge_model = UNSET
        elif isinstance(self.merge_model, LLMEnum):
            merge_model = self.merge_model.value
        else:
            merge_model = self.merge_model

        preview = self.preview

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "task": task,
            }
        )
        if merge_on_left is not UNSET:
            field_dict["merge_on_left"] = merge_on_left
        if merge_on_right is not UNSET:
            field_dict["merge_on_right"] = merge_on_right
        if merge_model is not UNSET:
            field_dict["merge_model"] = merge_model
        if preview is not UNSET:
            field_dict["preview"] = preview

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        task = d.pop("task")

        def _parse_merge_on_left(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        merge_on_left = _parse_merge_on_left(d.pop("merge_on_left", UNSET))

        def _parse_merge_on_right(data: object) -> None | str | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(None | str | Unset, data)

        merge_on_right = _parse_merge_on_right(d.pop("merge_on_right", UNSET))

        def _parse_merge_model(data: object) -> LLMEnum | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                merge_model_type_0 = LLMEnum(data)

                return merge_model_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(LLMEnum | None | Unset, data)

        merge_model = _parse_merge_model(d.pop("merge_model", UNSET))

        preview = d.pop("preview", UNSET)

        deep_merge_public_params = cls(
            task=task,
            merge_on_left=merge_on_left,
            merge_on_right=merge_on_right,
            merge_model=merge_model,
            preview=preview,
        )

        deep_merge_public_params.additional_properties = d
        return deep_merge_public_params

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
