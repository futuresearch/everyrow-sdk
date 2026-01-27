from enum import Enum


class DocumentQueryTool(str, Enum):
    QUERYDOCUMENTGEMINIURLCONTEXT = "QueryDocumentGeminiUrlContext"
    QUERYDOCUMENTLLM = "QueryDocumentLLM"
    QUERYDOCUMENTOMNI = "QueryDocumentOmni"
    QUERYDOCUMENTTEXT = "QueryDocumentText"
    QUERYDOCUMENTWITHCITATIONSLLM = "QueryDocumentWithCitationsLLM"

    def __str__(self) -> str:
        return str(self.value)
