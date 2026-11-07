import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.config import Settings
from app.rag_engine import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    ConfigurationError,
    EmptyKnowledgeBaseError,
    build_embeddings,
    build_llm,
    format_context,
    to_sources,
)
from tests.conftest import FAKE_ANSWER

CHUNK = "Il gatto dorme sul divano."


def _fill(engine):
    engine.add_documents(
        [
            Document(page_content=CHUNK, metadata={"source": "gatti.txt"}),
            Document(page_content="Altro testo completamente diverso.", metadata={"source": "altro.txt"}),
        ],
        ids=["a", "b"],
    )


def test_format_context_numbers_chunks_and_shows_page():
    docs = [
        Document(page_content="Testo A", metadata={"source": "a.pdf", "page": 3}),
        Document(page_content="Testo B", metadata={"source": "b.txt"}),
    ]
    out = format_context(docs)
    assert "[1] (Fonte: a.pdf, pagina 3)" in out
    assert "[2] (Fonte: b.txt)" in out


def test_to_sources_converts_metadata():
    docs = [Document(page_content="x", metadata={"source": "f.pdf", "page": 2.0}, id="abc")]
    source = to_sources(docs)[0]
    assert (source.source, source.page, source.chunk_id, source.content) == ("f.pdf", 2, "abc", "x")


def test_prompt_contains_not_found_rule_and_injection_guard():
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "non attendibile" in SYSTEM_PROMPT


def test_build_llm_and_embeddings_require_api_key():
    no_key = Settings(_env_file=None, openai_api_key="")
    with pytest.raises(ConfigurationError):
        build_llm(no_key)
    with pytest.raises(ConfigurationError):
        build_embeddings(no_key)


def test_query_on_empty_knowledge_base_raises(engine):
    with pytest.raises(EmptyKnowledgeBaseError):
        asyncio.run(engine.aquery("ciao"))


def test_query_returns_answer_and_sources(engine):
    _fill(engine)
    # Con embeddings deterministici, la stessa frase produce lo stesso vettore: il frammento esatto è il più vicino.
    answer, sources = asyncio.run(engine.aquery(CHUNK, top_k=1))
    assert answer == FAKE_ANSWER
    assert len(sources) == 1
    assert sources[0].source == "gatti.txt"


def test_context_and_question_are_passed_to_the_chain(engine):
    _fill(engine)
    captured = {}

    def fake_chain(inputs):
        captured.update(inputs)
        return "ok"

    engine.chain = RunnableLambda(fake_chain)
    asyncio.run(engine.aquery(CHUNK, top_k=1))
    assert "[1] (Fonte: gatti.txt)" in captured["context"]
    assert captured["question"] == CHUNK


def test_no_results_returns_not_found_message(engine, monkeypatch):
    _fill(engine)
    monkeypatch.setattr(engine.vectorstore, "similarity_search", lambda *a, **k: [])
    answer, sources = asyncio.run(engine.aquery("qualsiasi cosa"))
    assert answer == NOT_FOUND_MESSAGE
    assert sources == []
