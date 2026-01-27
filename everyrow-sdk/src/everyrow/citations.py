from copy import deepcopy
from typing import Any

from everyrow.generated.models import (
    ArtifactGroupRecord,
    AuxDataSourceBank,
    StandaloneArtifactRecord,
)
from everyrow.generated.types import Unset


def _render_citations(
    data: dict[str, Any], source_bank: AuxDataSourceBank
) -> dict[str, Any]:
    result = deepcopy(data)
    for source_id, source_data in source_bank.to_dict().items():
        for key, value in result.items():
            if isinstance(value, str):
                result[key] = value.replace(source_id, source_data["url"])
            else:
                result[key] = value
    return result


def render_citations_standalone(artifact: StandaloneArtifactRecord):
    if isinstance(artifact.aux_data, Unset) or isinstance(
        artifact.aux_data.source_bank, Unset
    ):
        return artifact
    source_bank = (
        artifact.aux_data.source_bank
    )  # create reference simply to make the type checker happy before deepcopy.
    artifact = deepcopy(artifact)
    artifact.data = _render_citations(artifact.data, source_bank)
    return artifact


def render_citations_group(artifact: ArtifactGroupRecord) -> ArtifactGroupRecord:
    artifact = deepcopy(artifact)
    new_artifacts = []
    for artifact_item in artifact.artifacts:
        if isinstance(artifact_item, StandaloneArtifactRecord):
            item_to_add = render_citations_standalone(artifact_item)
        elif isinstance(artifact_item, ArtifactGroupRecord):
            item_to_add = render_citations_group(artifact_item)
        else:
            item_to_add = artifact_item
        new_artifacts.append(item_to_add)
    artifact.artifacts = new_artifacts
    return artifact
