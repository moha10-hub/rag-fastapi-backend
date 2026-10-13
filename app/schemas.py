"""Modelli Pydantic per richieste e risposte delle API."""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    question: str = Field(..., max_length=2000, description="Domanda da porre ai documenti indicizzati")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Numero di frammenti da recuperare (default: RETRIEVER_K)",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("La domanda non può essere vuota")
        return value


class SourceDocument(BaseModel):
    source: str = Field(..., description="Nome del file di origine")
    page: Optional[int] = Field(default=None, description="Numero di pagina (solo PDF, base 1)")
    chunk_id: Optional[str] = Field(default=None, description="Identificativo del frammento nel vector store")
    content: str = Field(..., description="Testo del frammento usato come contesto")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument]


class UploadResponse(BaseModel):
    filename: str
    file_type: str
    chunks_added: int
    message: str


class HealthResponse(BaseModel):
    status: str
    engine_ready: bool
    indexed_chunks: Optional[int] = None
    llm_model: str
    embedding_provider: str
