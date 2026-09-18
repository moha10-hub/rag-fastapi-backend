# 🚀 Production-Ready RAG FastAPI Backend

Backend asincrono e modulare sviluppato con **FastAPI** e integrato con una pipeline **RAG (Retrieval-Augmented Generation)**. Il progetto è progettato per simulare standard di produzione aziendale (Enterprise-ready), con particolare attenzione alla validazione dei dati, alle performance asincrone e alla conformità dei modelli.

---

## 🛠️ Tech Stack & Architecture
* **Framework:** FastAPI (Async/Await, Uvicorn)
* **Validazione Dati:** Pydantic V2 (Modelli tipizzati per request/response)
* **Pipeline RAG:** Modello ibrido simulato (Hybrid Retrieval: BM25 + Dense Vectors + Reranking)
* **Architettura:** Pattern pulito e modulare (`app/main.py`, `app/services.py`, `app/schemas.py`)

---

## 📂 Struttura del Progetto
```text
rag-fastapi-backend/
│
├── app/
│   ├── __init__.py          # Inizializzatore pacchetto Python
│   ├── main.py              # Entrypoint FastAPI, routing e gestione errori
│   ├── schemas.py           # Schemi e validazione Pydantic
│   └── services.py          # Logica di retrieval ibrido e generazione basata su contesto
│
├── requirements.txt         # Dipendenze di progetto
└── README.md                # Documentazione
