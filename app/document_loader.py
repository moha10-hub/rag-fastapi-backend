"""Caricamento, parsing e chunking dei documenti (PDF/TXT) e inserimento nel vector store."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings

if TYPE_CHECKING:
    from app.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class DocumentProcessingError(Exception):
    """Errore generico durante la lettura o l'elaborazione di un documento."""


class UnsupportedFileTypeError(DocumentProcessingError):
    """Estensione del file non supportata."""


class EmptyDocumentError(DocumentProcessingError):
    """Il documento non contiene testo estraibile."""


def _load_pdf(path: Path, display_name: str) -> List[Document]:
    try:
        pages = PyPDFLoader(str(path)).load()
    except Exception as exc:  # pypdf solleva eccezioni diverse (file corrotto, cifrato, ...)
        raise DocumentProcessingError(f"Impossibile leggere il PDF '{display_name}': {exc}") from exc

    documents: List[Document] = []
    for index, page in enumerate(pages):
        text = page.page_content.strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": display_name, "page": index + 1},
            )
        )
    return documents


def _load_txt(path: Path, display_name: str) -> List[Document]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": display_name})]


def load_documents(path: Path, display_name: str, extension: str) -> List[Document]:
    """Legge il file e restituisce uno o più Document (uno per pagina nei PDF)."""
    extension = extension.lower()
    if extension == ".pdf":
        documents = _load_pdf(path, display_name)
    elif extension == ".txt":
        documents = _load_txt(path, display_name)
    else:
        raise UnsupportedFileTypeError(
            f"Formato '{extension}' non supportato. Formati ammessi: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if not documents:
        raise EmptyDocumentError(
            f"Nessun testo estraibile da '{display_name}'. Se è un PDF scansionato serve prima un OCR."
        )
    return documents


def split_documents(documents: List[Document], settings: Settings) -> List[Document]:
    """Divide i documenti in frammenti con RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(documents)


def compute_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def ingest_file(path: Path, display_name: str, engine: "RAGEngine", settings: Settings) -> int:
    """Pipeline completa: lettura -> chunking -> salvataggio su ChromaDB.

    Gli ID dei frammenti derivano dall'hash del contenuto del file: ricaricare lo stesso file
    aggiorna i frammenti esistenti invece di duplicarli.
    Restituisce il numero di frammenti salvati.
    """
    extension = Path(display_name).suffix.lower()
    documents = load_documents(path, display_name, extension)
    chunks = split_documents(documents, settings)
    if not chunks:
        raise EmptyDocumentError(f"Nessun frammento generato da '{display_name}'.")

    file_hash = compute_file_hash(path)
    ids: List[str] = []
    for index, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = file_hash
        chunk.metadata["chunk_index"] = index
        ids.append(f"{file_hash}-{index}")

    engine.add_documents(chunks, ids)
    logger.info("Indicizzato '%s': %d pagine/sezioni, %d frammenti", display_name, len(documents), len(chunks))
    return len(chunks)
