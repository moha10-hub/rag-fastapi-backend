"""Fixture condivise: motore RAG con embeddings e LLM finti, senza rete né chiavi API."""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.rag_engine import RAGEngine

FAKE_ANSWER = "Risposta di prova [1]"


@pytest.fixture
def settings(tmp_path):
    # _env_file=None: i test non leggono mai il tuo .env reale.
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        vector_db_dir=tmp_path / "chroma",
        collection_name="test_collection",
        chunk_size=200,
        chunk_overlap=20,
    )


@pytest.fixture
def engine(settings):
    return RAGEngine(
        settings,
        embeddings=DeterministicFakeEmbedding(size=32),
        llm=FakeListChatModel(responses=[FAKE_ANSWER]),
    )
