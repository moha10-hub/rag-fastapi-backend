class RAGService:
    @staticmethod
    def hybrid_retrieve(query: str, top_k: int = 3):
        """
        Simula un recupero ibrido (Keyword BM25 + Dense Vectors + Reranker).
        In un contesto reale, qui collegheresti Qdrant, Chroma o PostgreSQL/pgvector.
        """
        mock_kb = [
            {"chunk_id": "doc_101", "content": "L'architettura RAG in produzione richiede un sistema di reranking per migliorare la precisione del contesto.", "score": 0.95},
            {"chunk_id": "doc_102", "content": "FastAPI garantisce performance asincrone elevate, rendendolo ideale per servire API basate su LLM e microservizi AI.", "score": 0.89},
            {"chunk_id": "doc_103", "content": "La conformità all'EU AI Act richiede trasparenza e tracciabilità nelle risposte generate dai modelli linguistici.", "score": 0.82},
        ]
        return mock_kb[:top_k]

    @classmethod
    def generate_answer(cls, query: str, retrieved_docs: list) -> str:
        """
        Simula la generazione della risposta da parte del modello basandosi sul contesto recuperato (Grounding).
        """
        return f"Risposta generata analizzando il contesto sicuro: '{query}' basandosi su {len(retrieved_docs)} fonti verificate."
