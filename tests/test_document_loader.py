import pytest
from langchain_core.documents import Document

from app import document_loader
from app.document_loader import (
    DocumentProcessingError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
    compute_file_hash,
    ingest_file,
    load_documents,
    split_documents,
)


def test_load_txt_utf8(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("perché è un test", encoding="utf-8")
    docs = load_documents(path, "a.txt", ".txt")
    assert len(docs) == 1
    assert docs[0].metadata == {"source": "a.txt"}
    assert "perché" in docs[0].page_content


def test_load_txt_falls_back_to_latin1(tmp_path):
    path = tmp_path / "b.txt"
    path.write_bytes("perché".encode("latin-1"))
    docs = load_documents(path, "b.txt", ".txt")
    assert "perché" in docs[0].page_content


def test_blank_txt_raises_empty_document_error(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("   \n  ")
    with pytest.raises(EmptyDocumentError):
        load_documents(path, "c.txt", ".txt")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "d.docx"
    path.write_bytes(b"x")
    with pytest.raises(UnsupportedFileTypeError):
        load_documents(path, "d.docx", ".docx")


def test_pdf_pages_are_one_based_and_blank_pages_skipped(tmp_path, monkeypatch):
    class FakePDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            return [
                Document(page_content="Prima pagina", metadata={}),
                Document(page_content="   ", metadata={}),
                Document(page_content="Terza pagina", metadata={}),
            ]

    monkeypatch.setattr(document_loader, "PyPDFLoader", FakePDFLoader)
    docs = load_documents(tmp_path / "x.pdf", "x.pdf", ".pdf")
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert all(d.metadata["source"] == "x.pdf" for d in docs)


def test_pdf_read_error_becomes_processing_error(tmp_path, monkeypatch):
    class BrokenPDFLoader:
        def __init__(self, path):
            pass

        def load(self):
            raise ValueError("file corrotto")

    monkeypatch.setattr(document_loader, "PyPDFLoader", BrokenPDFLoader)
    with pytest.raises(DocumentProcessingError):
        load_documents(tmp_path / "x.pdf", "x.pdf", ".pdf")


def test_split_documents_respects_chunk_size(settings):
    docs = [Document(page_content="parola " * 300, metadata={"source": "a.txt"})]
    chunks = split_documents(docs, settings)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


def test_file_hash_depends_only_on_content(tmp_path):
    a, b, c = tmp_path / "a.txt", tmp_path / "b.txt", tmp_path / "c.txt"
    a.write_text("stesso contenuto")
    b.write_text("stesso contenuto")
    c.write_text("contenuto diverso")
    assert compute_file_hash(a) == compute_file_hash(b)
    assert compute_file_hash(a) != compute_file_hash(c)


def test_ingest_same_file_twice_does_not_duplicate_chunks(tmp_path, engine, settings):
    path = tmp_path / "doc.txt"
    path.write_text("parola " * 300)

    first = ingest_file(path, "doc.txt", engine, settings)
    count_after_first = engine.count_chunks()
    second = ingest_file(path, "doc.txt", engine, settings)

    assert first == second == count_after_first
    assert engine.count_chunks() == count_after_first
