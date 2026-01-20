from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..models.dedupe_mode import DedupeMode
from ..models.embedding_models import EmbeddingModels
from ..models.llm_enum import LLMEnum
from ..types import UNSET, Unset

T = TypeVar("T", bound="DedupeFullParams")


@_attrs_define
class DedupeFullParams:
    """Full parameters for the deduplication service.

    Attributes:
        equivalence_relation (str): Description of what makes items equivalent
        llm (LLMEnum | Unset):
        chunk_size (int | Unset): Maximum number of items to process in a single LLM call Default: 25.
        mode (DedupeMode | Unset):
        preview (bool | Unset): When true, process only the first few items Default: False.
        embedding_model (EmbeddingModels | Unset):
        validate_groups (bool | Unset): Validate equivalence classes and split incorrectly merged groups before
            selecting representatives Default: False.
        use_clustering (bool | Unset): When true, cluster items by embedding similarity and only compare neighboring
            clusters. When false, use sequential chunking and compare all chunks (O(n²)) Default: True.
        use_column_selection (bool | Unset): When true, use LLM to select relevant columns for deduplication, reducing
            noise from irrelevant fields like IDs or timestamps Default: True.
        early_stopping_threshold (int | None | Unset): Stop cross-chunk comparisons for a row after this many
            consecutive comparisons with no matches. Only applies when use_clustering=True. When use_clustering=False, all
            chunks are compared exhaustively. None disables early stopping. Default: 5.
        validation_model (LLMEnum | None | Unset): Model for group validation (defaults to llm if not set) Default:
            LLMEnum.GEMINI_3_FLASH_MINIMAL.
        representative_model (LLMEnum | None | Unset): Model for choosing best representative (defaults to llm if not
            set) Default: LLMEnum.GEMINI_2_5_FLASH.
        select_representative (bool | Unset): When true, use LLM to select the best representative from each equivalence
            class. When false, no selection is made. Default: True.
    """

    equivalence_relation: str
    llm: LLMEnum | Unset = UNSET
    chunk_size: int | Unset = 25
    mode: DedupeMode | Unset = UNSET
    preview: bool | Unset = False
    embedding_model: EmbeddingModels | Unset = UNSET
    validate_groups: bool | Unset = False
    use_clustering: bool | Unset = True
    use_column_selection: bool | Unset = True
    early_stopping_threshold: int | None | Unset = 5
    validation_model: LLMEnum | None | Unset = LLMEnum.GEMINI_3_FLASH_MINIMAL
    representative_model: LLMEnum | None | Unset = LLMEnum.GEMINI_2_5_FLASH
    select_representative: bool | Unset = True
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        equivalence_relation = self.equivalence_relation

        llm: str | Unset = UNSET
        if not isinstance(self.llm, Unset):
            llm = self.llm.value

        chunk_size = self.chunk_size

        mode: str | Unset = UNSET
        if not isinstance(self.mode, Unset):
            mode = self.mode.value

        preview = self.preview

        embedding_model: str | Unset = UNSET
        if not isinstance(self.embedding_model, Unset):
            embedding_model = self.embedding_model.value

        validate_groups = self.validate_groups

        use_clustering = self.use_clustering

        use_column_selection = self.use_column_selection

        early_stopping_threshold: int | None | Unset
        if isinstance(self.early_stopping_threshold, Unset):
            early_stopping_threshold = UNSET
        else:
            early_stopping_threshold = self.early_stopping_threshold

        validation_model: None | str | Unset
        if isinstance(self.validation_model, Unset):
            validation_model = UNSET
        elif isinstance(self.validation_model, LLMEnum):
            validation_model = self.validation_model.value
        else:
            validation_model = self.validation_model

        representative_model: None | str | Unset
        if isinstance(self.representative_model, Unset):
            representative_model = UNSET
        elif isinstance(self.representative_model, LLMEnum):
            representative_model = self.representative_model.value
        else:
            representative_model = self.representative_model

        select_representative = self.select_representative

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update(
            {
                "equivalence_relation": equivalence_relation,
            }
        )
        if llm is not UNSET:
            field_dict["llm"] = llm
        if chunk_size is not UNSET:
            field_dict["chunk_size"] = chunk_size
        if mode is not UNSET:
            field_dict["mode"] = mode
        if preview is not UNSET:
            field_dict["preview"] = preview
        if embedding_model is not UNSET:
            field_dict["embedding_model"] = embedding_model
        if validate_groups is not UNSET:
            field_dict["validate_groups"] = validate_groups
        if use_clustering is not UNSET:
            field_dict["use_clustering"] = use_clustering
        if use_column_selection is not UNSET:
            field_dict["use_column_selection"] = use_column_selection
        if early_stopping_threshold is not UNSET:
            field_dict["early_stopping_threshold"] = early_stopping_threshold
        if validation_model is not UNSET:
            field_dict["validation_model"] = validation_model
        if representative_model is not UNSET:
            field_dict["representative_model"] = representative_model
        if select_representative is not UNSET:
            field_dict["select_representative"] = select_representative

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        d = dict(src_dict)
        equivalence_relation = d.pop("equivalence_relation")

        _llm = d.pop("llm", UNSET)
        llm: LLMEnum | Unset
        if isinstance(_llm, Unset):
            llm = UNSET
        else:
            llm = LLMEnum(_llm)

        chunk_size = d.pop("chunk_size", UNSET)

        _mode = d.pop("mode", UNSET)
        mode: DedupeMode | Unset
        if isinstance(_mode, Unset):
            mode = UNSET
        else:
            mode = DedupeMode(_mode)

        preview = d.pop("preview", UNSET)

        _embedding_model = d.pop("embedding_model", UNSET)
        embedding_model: EmbeddingModels | Unset
        if isinstance(_embedding_model, Unset):
            embedding_model = UNSET
        else:
            embedding_model = EmbeddingModels(_embedding_model)

        validate_groups = d.pop("validate_groups", UNSET)

        use_clustering = d.pop("use_clustering", UNSET)

        use_column_selection = d.pop("use_column_selection", UNSET)

        def _parse_early_stopping_threshold(data: object) -> int | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(int | None | Unset, data)

        early_stopping_threshold = _parse_early_stopping_threshold(d.pop("early_stopping_threshold", UNSET))

        def _parse_validation_model(data: object) -> LLMEnum | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                validation_model_type_0 = LLMEnum(data)

                return validation_model_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(LLMEnum | None | Unset, data)

        validation_model = _parse_validation_model(d.pop("validation_model", UNSET))

        def _parse_representative_model(data: object) -> LLMEnum | None | Unset:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, str):
                    raise TypeError()
                representative_model_type_0 = LLMEnum(data)

                return representative_model_type_0
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            return cast(LLMEnum | None | Unset, data)

        representative_model = _parse_representative_model(d.pop("representative_model", UNSET))

        select_representative = d.pop("select_representative", UNSET)

        dedupe_full_params = cls(
            equivalence_relation=equivalence_relation,
            llm=llm,
            chunk_size=chunk_size,
            mode=mode,
            preview=preview,
            embedding_model=embedding_model,
            validate_groups=validate_groups,
            use_clustering=use_clustering,
            use_column_selection=use_column_selection,
            early_stopping_threshold=early_stopping_threshold,
            validation_model=validation_model,
            representative_model=representative_model,
            select_representative=select_representative,
        )

        dedupe_full_params.additional_properties = d
        return dedupe_full_params

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
