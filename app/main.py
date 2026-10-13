"""Applicazione FastAPI: /health, /upload (ingestion) e /query (RAG con fonti)."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.document_loader import SUPPORTED_EXTENSIONS, DocumentProcessingError, ingest_file
from app.rag_engine import (
    ConfigurationError,
    EmptyKnowledgeBaseError,
    RAGEngine,
    get_engine,
    is_engine_ready,
)
from app.schemas import HealthResponse, QueryRequest, QueryResponse, UploadResponse

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app.main")


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Inizializza il motore all'avvio. Se manca la configurazione l'app parte lo stesso:
    # /health lo segnala e le rotte rispondono 503 finché non viene corretta.
    try:
        await run_in_threadpool(get_engine)
    except ConfigurationError as exc:
        logger.error("Motore RAG non inizializzato: %s", exc)
    except Exception:
        logger.exception("Errore imprevisto durante l'inizializzazione del motore RAG")
    yield


app = FastAPI(
    title=settings.app_name,
    description="Backend RAG: carica PDF/TXT e interroga i documenti con citazione delle fonti.",
    version="1.0.0",
    lifespan=lifespan,
)

_origins = settings.cors_origins_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials="*" not in _origins,  # con "*" i browser rifiutano le credenziali
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Gestione delle eccezioni
# --------------------------------------------------------------------------- #
def _error(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


@app.exception_handler(ConfigurationError)
async def configuration_error_handler(_: Request, exc: ConfigurationError) -> JSONResponse:
    logger.error("Errore di configurazione: %s", exc)
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


@app.exception_handler(EmptyKnowledgeBaseError)
async def empty_knowledge_base_handler(_: Request, exc: EmptyKnowledgeBaseError) -> JSONResponse:
    return _error(status.HTTP_409_CONFLICT, str(exc))


@app.exception_handler(DocumentProcessingError)
async def document_processing_handler(_: Request, exc: DocumentProcessingError) -> JSONResponse:
    logger.warning("Documento non elaborabile: %s", exc)
    return _error(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))


@app.exception_handler(openai.RateLimitError)
async def openai_rate_limit_handler(_: Request, exc: openai.RateLimitError) -> JSONResponse:
    logger.warning("OpenAI rate limit / quota: %s", exc)
    return _error(status.HTTP_429_TOO_MANY_REQUESTS, "Limite di richieste o quota OpenAI raggiunti. Riprova più tardi.")


@app.exception_handler(openai.AuthenticationError)
async def openai_auth_handler(_: Request, exc: openai.AuthenticationError) -> JSONResponse:
    logger.error("OpenAI ha rifiutato le credenziali: %s", exc)
    return _error(status.HTTP_502_BAD_GATEWAY, "OpenAI ha rifiutato la chiave API configurata.")


@app.exception_handler(openai.APIConnectionError)
async def openai_connection_handler(_: Request, exc: openai.APIConnectionError) -> JSONResponse:
    logger.error("Connessione a OpenAI fallita: %s", exc)
    return _error(status.HTTP_503_SERVICE_UNAVAILABLE, "Impossibile contattare OpenAI. Riprova più tardi.")


@app.exception_handler(openai.OpenAIError)
async def openai_generic_handler(_: Request, exc: openai.OpenAIError) -> JSONResponse:
    logger.error("Errore OpenAI: %s", exc)
    return _error(status.HTTP_502_BAD_GATEWAY, "Errore del servizio OpenAI.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Errore non gestito", exc_info=exc)
    return _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Errore interno del server.")


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
async def _save_upload_to_temp(file: UploadFile, extension: str, max_bytes: int) -> Path:
    """Salva l'upload su un file temporaneo leggendo a blocchi e rispettando il limite di dimensione."""
    descriptor, name = tempfile.mkstemp(suffix=extension)
    path = Path(name)
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File troppo grande: il limite è {settings.max_upload_mb} MB.",
                    )
                output.write(block)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Il file caricato è vuoto.")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


# --------------------------------------------------------------------------- #
# Rotte
# --------------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    ready = is_engine_ready()
    indexed_chunks = None
    if ready:
        try:
            indexed_chunks = await run_in_threadpool(get_engine().count_chunks)
        except Exception:
            logger.exception("Impossibile leggere il numero di frammenti indicizzati")
    return HealthResponse(
        status="ok",
        engine_ready=ready,
        indexed_chunks=indexed_chunks,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
    )


@app.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED, tags=["rag"])
async def upload_document(
    file: UploadFile = File(..., description="File PDF o TXT da indicizzare"),
    engine: RAGEngine = Depends(get_engine),
) -> UploadResponse:
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Nome del file mancante.")

    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Formato non supportato. Formati ammessi: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )

    temp_path = None
    try:
        temp_path = await _save_upload_to_temp(file, extension, settings.max_upload_bytes)
        chunks_added = await run_in_threadpool(ingest_file, temp_path, original_name, engine, settings)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        await file.close()

    return UploadResponse(
        filename=original_name,
        file_type=extension.lstrip("."),
        chunks_added=chunks_added,
        message=f"File '{original_name}' indicizzato in {chunks_added} frammenti.",
    )


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query_documents(
    request: QueryRequest,
    engine: RAGEngine = Depends(get_engine),
) -> QueryResponse:
    answer, sources = await engine.aquery(request.question, request.top_k)
    return QueryResponse(question=request.question, answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
