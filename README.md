# Multi-Format Hybrid RAG System

Ce projet implémente un système RAG (Retrieval Augmented Generation)
permettant d’indexer, synchroniser et interroger des documents
multiformats (PDF, TXT, PPTX, XLSX, CSV).

Statut: IN PROGRESS...

## Architecture
- FastAPI → API backend
- PostgreSQL + pgvector → stockage embeddings
- LangChain → pipeline RAG
- Mistral → embeddings + génération
- Docker → orchestration
- pgAdmin

## Ingestion & Synchronisation incrémentale
Le système implémente une synchronisation du type:
- Scan du dossier documentaire
- Détection des nouveaux fichiers
- Détection des modifications via hash 
- Indexation uniquement des changements 


Reste à réaliser:
- ameliorer le retrieval/génération [ ]
- evaluer le système (framework: deepeval, RAGAS, autre) [ ]

## Optimisations futures
- Cache Redis pour accélérer les requêtes répétitives
- Intégration ColPali pour indexation visuelle (graphiques, schémas)
- Note : Synchronisation temps réel non prioritaire — le système actuel suffit pour des collections peu changeantes


- **Stack**: FastAPI, PostgreSQL (pgvector), LangChain, MistralAI, PgAdmin, Docker, Streamlit.

## Installation
1. `docker-compose up -d`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
...
