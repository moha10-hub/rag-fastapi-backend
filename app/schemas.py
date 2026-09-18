from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="La domanda dell'utente da ricercare nella knowledge base")
    top_k: Optional[int] = Field(3, description="Numero di chunk da recuperare per il contesto")

class SourceDocument(BaseModel):
    chunk_id: str
    content: str
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_sources: List[SourceDocument]
    model_used: str = "rag-hybrid-mock-v1"
