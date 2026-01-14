from enum import Enum


class EmbeddingModels(str, Enum):
    TEXT_EMBEDDING_3_LARGE = "text-embedding-3-large"
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

    def __str__(self) -> str:
        return str(self.value)
