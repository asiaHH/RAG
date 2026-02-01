# Multi-Format Hybrid RAG System

Statut: IN PROGRESS...

Système RAG utilisant Mistral, LangChain et pgvector.

## Features
- **Hybrid Search**: Combinaison de recherche sémantique et BM25.
- **Auto-Sync**: Ingestion sélective (PDF, TXT, MD) depuis un dossier local.
- **Stack**: FastAPI, PostgreSQL (pgvector), Docker, Streamlit.

## Installation
1. `docker-compose up -d`
2. `pip install -r requirements.txt`
3. `streamlit run streamlit_app.py`
