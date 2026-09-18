from fastapi import FastAPI, HTTPException, status
from app.schemas import QueryRequest, QueryResponse, SourceDocument
from app.services import RAGService

app = FastAPI(
    title="Production-Ready RAG API",
    description="Backend asincrono con FastAPI per interrogazione RAG ibrida orientata ai requisiti enterprise e conformità.",
    version="1.0.0"
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Endpoint di controllo dello stato del servizio."""
    return {"status": "healthy", "service": "rag-backend"}

@app.post("/api/v1/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_knowledge_base(payload: QueryRequest):
    """
    Endpoint principale per l'interrogazione della Knowledge Base.
    """
    try:
        retrieved_raw = RAGService.hybrid_retrieve(query=payload.query, top_k=payload.top_k)
        
        sources = [
            SourceDocument(chunk_id=doc["chunk_id"], content=doc["content"], score=doc["score"])
            for doc in retrieved_raw
        ]
        
        answer = RAGService.generate_answer(query=payload.query, retrieved_docs=retrieved_raw)
        
        return QueryResponse(
            query=payload.query,
            answer=answer,
            retrieved_sources=sources
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore interno durante l'elaborazione della pipeline RAG: {str(e)}"
        )
