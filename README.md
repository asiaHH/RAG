# Multi-Format Hybrid RAG System

Ce projet implémente un système RAG (Retrieval Augmented Generation)
permettant d’indexer, synchroniser et interroger des documents
multiformats (PDF, DOCX, TXT, PPTX, XLSX, CSV).

Statut: IN PROGRESS...

---

## Stack technique

| Composant | Rôle |
|-----------|------|
| FastAPI | Backend API |
| PostgreSQL + pgvector | Stockage des embeddings |
| LangChain | Pipeline RAG |
| MistralAI | Embeddings + génération |
| Gemini Flash | LLM juge (évaluation) |
| DeepEval | Framework d'évaluation |
| Streamlit | Interface utilisateur |
| Docker + pgAdmin | Orchestration & administration |

---

## Fonctionnalités implémentées

### Ingestion & Synchronisation incrémentale
- Scan du dossier documentaire
- Détection des nouveaux fichiers
- Détection des modifications via hash 
- Indexation uniquement des changements 
- Formats supportés : PDF, DOCX, TXT, PPTX, XLSX, CSV

### Nettoyage des données
Principe : nettoyer le minimum nécessaire, chaque transformation supprime de l'information.

- Suppression des caractères NUL
- Suppression des caractères invisibles (Zero-Width)
- Normalisation des espaces multiples

### Évaluation (v1)
- Génération de dataset via Mistral (questions variées, dimensions configurables)
- Round-trip check : vérifie que les questions positives retrouvent leur chunk source
- Ratios positifs/négatifs paramétrables
- LLM juge : Gemini Flash (pour éviter le self-enhancement bias)

---

## Roadmap

### Version actuelle (v1) — Retrieval sémantique seul
- [ ] Évaluer le système v1 avec le dataset généré
- [ ] Commit sur GitHub

### Version 2 — Amélioration du retrieval
- [ ] Ajouter BM25 (recherche hybride sémantique + lexicale)
- [ ] Ré-évaluer et comparer avec v1
- [ ] Commit sur GitHub

### Évaluation — Évolution prévue
Remplacement partiel de DeepEval pour limiter la consommation de quotas LLM :
- Calcul des métriques de retrieval en Python pur (pas de LLM)
- DeepEval conservé uniquement pour la génération du dataset
- Gemini Flash reste le LLM juge pour la partie génération

---

## Installation
```bash
docker-compose up -d
pip install -r requirements.txt
streamlit run src/app.py
```

### Lancer l'application

```bash
# Activer le virtualenv
source venv/bin/activate

# Lancer l'API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Lancer l'interface
streamlit run src/app.py
```

PgAdmin : http://localhost:8080

---

## Évaluation

### Installation des dépendances

```bash
pip install -r evaluation/requirements-eval.txt
```

### Configuration

Obtenir une clé API Google AI via [Google AI Studio](https://aistudio.google.com/) et l'ajouter au `.env` :

```
GOOGLE_API_KEY="votre_clé_api_ici"
```

### Générer un dataset

```bash
python -m evaluation.dataset.generate_dataset --ratio 0.7 --n_questions 10
```

Le dataset contient 10 questions représentatives — volume suffisant pour itérer rapidement et maîtriser les coûts API.

### Lancer l'évaluation

```bash
python -m evaluation.run_eval

# Avec un dataset personnalisé
python -m evaluation.run_eval --dataset generated_dataset_ratio_0.7.json
```

### Notes sur le biais d'évaluation

**Self-enhancement bias** : un LLM tend à favoriser ses propres outputs quand il s'évalue lui-même. Solution retenue : Mistral génère les réponses, Gemini Flash juge.