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
- Vérification des doublons par requête SQL 
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
- [X] Évaluer le système v1 avec le dataset généré
- [X] Erreur retrouvé dans l'ingestion = doublons à corriger 
- [ ] Ré-évaluer 



### Version 2 — Amélioration du retrieval
- [ ] Ajouter BM25 (recherche hybride sémantique + lexicale)
- [ ] Ré-évaluer et comparer avec v1

### Évaluation — Évolution prévue
Remplacement partiel de DeepEval pour limiter la consommation de quotas LLM :
- [X] Calcul des métriques de retrieval en Python pur (pas de LLM)
- [X] DeepEval conservé uniquement pour la génération du dataset
- [X] Gemini Flash reste le LLM juge pour la partie génération

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


# Évaluation retrieval uniquement
python -m evaluation.run_eval --retrieval

# Évaluation génération uniquement
python -m evaluation.run_eval --generation

# Les deux
python -m evaluation.run_eval



## Stratégie d'évaluation : le dataset

Le dataset génératif sert à deux niveaux distincts : valider l'évaluateur lui-même, puis avoir confiance dans les résultats obtenus.

### Niveau 1 — Valider l'évaluateur

Le dataset `generated_dataset_ratio_0.7.json` a été construit à partir du pipeline d'ingestion, sans annotation humaine. Le `ratio_0.7` signifie que 70% des questions ont leur `assigned_chunk_id` assigné de façon fiable.

### Niveau 2 — Confiance dans les résultats

Une fois vérifié que l'évaluateur se comporte correctement sur ce dataset connu, les scores obtenus reflètent fidèlement les performances réelles du retriever.


# Rapport d'Évaluation de la Génération (V1)
### Partie Retrieval:
Le Retrieval est à améliorer.


### Partie Génération:

L'évaluation a été réalisée à l'aide du framework DeepEval. 
Les données du dataset ont été générées par le modèle Mistral. Chaque (Question/Réponse/Contexte) à été soumis à une évalution via Gemini 2.5 Flash comme modèle de juge (LLM-as-a-judge). L'objectif était de mesurer la performance sur la Fidélité et la Pertinence des réponses.

*Objectif*: Utiliser le modèle Gemini pour détecter si Mistral a halluciné ou s'il a été imprécis.

*Résultat V1*: Avec des scores dépassant les 90%, le contenu généré par Mistral est plutôt fiable. 
Cependant, dans ce dataset il y avait 3 questions pièges (non pertinentes):
Mistral a échoué sur certains cas précis. Dans 1 des 3 questions pièges Mistral a déduit et inventé une réponse à partir du titre d'une figure alors qu'il n'y avait aucune information comme réponse attendue. 
Dans une autre question pièg, il a halluciné car le contexte fourni pour cette question contenait que quelque ligne d'informations. Mistral a donc utilisé ses connaissances internes au lieu de se limiter au contexte. 

Pourquoi Pass par DeepEval? 
Hypothèses: Sûrement une mauvaise interprétation, le juge a du considéré que comme Mistral cite le nom de la figure "Figure 2" qui est dans le texte il est fidèle. 
Et pour l'autre question l'answer relevancy est pass, surement parce que la réponse est pertinente même si fausse par rapport au corpus. 

Correction: Durcir le prompt!

-------------------------------------------------------------------
Métrique            Score Moyen     Seuil (Threshold)       État
-------------------------------------------------------------------
Faithfulness        0.93            0.80                    Succès
Answer Relevancy    0.90            0.75                    Succès
-------------------------------------------------------------------




