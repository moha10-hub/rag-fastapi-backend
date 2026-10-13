# RAG FastAPI Backend

Backend API to chat with your own documents (PDF and TXT) using Retrieval-Augmented Generation (RAG).

## Features
- `POST /upload`: upload a PDF or TXT file. The text is split into chunks and stored in ChromaDB (saved on disk).
- `POST /query`: ask a question. The most similar chunks are retrieved and an LLM (OpenAI `gpt-4o-mini`) answers using only that context.
- Every answer includes its **sources** (file name, page and the text used).
- `GET /health`: service status.
- Settings loaded from a `.env` file, CORS enabled, Docker ready.

## Tech stack
Python, FastAPI, LangChain (LCEL), ChromaDB, OpenAI API, Pydantic, Docker

## Project structure
```
app/
  main.py             # FastAPI app and routes
  config.py           # settings from .env
  schemas.py          # request/response models
  document_loader.py  # file loading, chunking, ingestion
  rag_engine.py       # embeddings, vector store, RAG chain
Dockerfile
requirements.txt
.env.example
```

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env          # then add your OPENAI_API_KEY
uvicorn app.main:app --reload
```
Interactive docs: http://127.0.0.1:8000/docs

## Example
```bash
curl -F "file=@document.pdf" http://localhost:8000/upload
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
     -d '{"question": "What is this document about?"}'
```

## Docker
```bash
docker build -t rag-backend .
docker run -p 8000:8000 --env-file .env -v rag_data:/app/chroma_db rag-backend
```
