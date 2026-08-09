# Multi-Format Hybrid RAG System

Ce projet implémente un système RAG (Retrieval Augmented Generation)
permettant d’indexer, synchroniser et interroger des documents
multiformats (PDF, DOCX, TXT, PPTX, XLSX, CSV).

PROJET EN COURS
---

## Stack technique

| Composant | Rôle |
|-----------|------|
| FastAPI | Backend API |
| PostgreSQL + pgvector | Stockage des embeddings |
| LangChain | Pipeline RAG |
| MistralAI | Embeddings + génération |
| Gemini Pro | LLM juge (évaluation) |
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
- LLM juge : Gemini Pro (pour éviter le self-enhancement bias)

---

## Roadmap

### Version (v1) — Retrieval sémantique seul
- [ ] Évaluer le système v1

### Version 2 — Amélioration du retrieval
- [X] Ajouter BM25 (recherche hybride sémantique + lexicale)
- [ ] Ré-évaluer et comparer avec v1

---

## Installation
```bash
docker-compose up -d
pip install -r requirements.txt
streamlit run src/app.py
```


## Lancer l'application

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
python -m evaluation.dataset.generate_dataset --ratio 0.7 --n_questions 30
```

### Lancer l'évaluation

```bash
python -m evaluation.run_eval

# Avec un dataset personnalisé
python -m evaluation.run_eval --dataset generated_dataset_ratio_0.7.json
```

#### Notes sur le biais d'évaluation

**Self-enhancement bias** : un LLM tend à favoriser ses propres outputs quand il s'évalue lui-même. Solution retenue : Mistral génère les réponses, Gemini Pro juge.


#### Évaluation retrieval uniquement
> python -m evaluation.run_eval --retrieval

#### Évaluation génération uniquement
> python -m evaluation.run_eval --generation

#### Les deux
> python -m evaluation.run_eval


#### Dataset

Le dataset `generated_dataset_ratio_0.7.json` a été construit à partir du pipeline d'ingestion, sans annotation humaine. Le `ratio_0.7` signifie que 70% des questions ont leur `assigned_chunk_id` assigné de façon fiable.


## Choix des outils d'évaluation

L'évaluation du système est volontairement séparée en deux approches distinctes, choisies selon la nature de ce qu'il y a à mesurer :

**Retrieval → métriques Python pures, sans dépendance.** Precision@K, Recall@K, MRR et Hit Rate@K se calculent via code, aucun jugement sémantique n'est nécessaire, donc aucun appel LLM.

**Génération → DeepEval (Faithfulness, Answer Relevancy).** Juger si une réponse en langage naturel est fidèle à son contexte ou pertinente par rapport à une question nécessite un jugement sémantique qu'aucune métrique ne peut capturer correctement, c'est le rôle d'un LLM-judge. J'avais déjà implémenté ce type d'évaluation en code pur lors d'un stage précédent; j'ai choisi DeepEval sur ce projet pour découvrir une approche outillée de l'évaluation par LLM-judge et comparer les deux façons de faire.


Choix: privilégier une méthode déterministe et gratuite quand le problème le permet (retrieval), et réserver le LLM-judge aux cas où un jugement sémantique est réellement incontournable (génération).


## Méthodologie d'évaluation

Le dataset d'évaluation contient deux types de questions, dans un ratio 70% pertinentes / 30% hors-corpus:

- **Questions pertinentes** : la réponse existe dans le corpus, associée à un chunk source précis (`assigned_chunk_id`).
- **Questions hors-corpus** : construites pour sembler plausibles mais sans la bonne réponse — elles permettront de tester la capacité du système à ne pas halluciner plutôt qu'à bien répondre.

Chaque type de question évalue un étage différent du pipeline, avec des métriques adaptées à ce qu'il y a réellement à mesurer :

| Étage | Sous-ensemble utilisé | Métriques | Ce qui est mesuré |
|-------|-----------------------|-----------|-------------------|
| **Retrieval** | Pertinentes uniquement | Precision@K, Recall@K, MRR, Hit Rate@K | Le retriever retrouve-t-il le bon chunk source ? |
| **Génération** | Pertinentes uniquement | Faithfulness, Answer Relevancy | La réponse générée est-elle fidèle au contexte et pertinente vis-à-vis de la question ? |
| **Abstention** | Hors-corpus uniquement | (prochainement) | Le système reconnaît-il correctement l'absence d'information plutôt que d'inventer une réponse ? |


**Pourquoi ne pas tout évaluer ensemble ?** Les métriques de retrieval supposent l'existence d'un document pertinent à retrouver, elles n'ont pas de sens sur une question conçue pour n'avoir aucune réponse dans le corpus. De même, Faithfulness et Answer Relevancy sont conçues pour juger la qualité d'une réponse factuelle, pas la pertinence d'un refus de répondre.

L'évaluation de la génération (DeepEval + Gemini) est réalisée sur un échantillon de 20 questions pertinentes (tirage aléatoire), plutôt que sur l'ensemble du dataset, en raison du coût en appels API du LLM-judge. L'évaluation du retrieval, gratuite et déterministe, est réalisée sur l'ensemble des questions pertinentes.


### Résultats — Retrieval (recherche sémantique)

<img width="450" height="257" alt="Capture d&#39;écran 2026-08-04 230715" src="https://github.com/user-attachments/assets/a5de6288-ebcb-4a04-a52c-9f67f5baa831" />

<img width="626" height="258" alt="Capture d&#39;écran 2026-08-04 231625" src="https://github.com/user-attachments/assets/f31bb63a-478f-48ec-8726-a6e3d6f360ea" />

### Résultats — Retrieval (recherche hybride)

<img width="615" height="257" alt="Capture d&#39;écran 2026-08-05 040453" src="https://github.com/user-attachments/assets/81280714-090f-40fd-a25c-b7d7851693c4" />

Sur ce dataset, l'hybride n'apporte pas d'amélioration mesurable, probablement parce que les questions générées sont sémantiquement proches de leur source. La comparaison se fera sur une évaluation humaine dans streamlit. 
### Ré-évaluation avec poids différent - Impact du poids BM25 dans la fusion hybride

| Config | Precision@5 | Recall@5 | MRR | Hit Rate@5 |
|--------|-------------|----------|-----|------------|
| Sémantique seul | 0.2000 | 1.0000 | 0.8767 | 1.0000 |
| Hybride (BM25 0.2 / Sém. 0.8) | 0.2000 | 1.0000 | 0.8708 | 1.0000 |
| Hybride (BM25 0.5 / Sém. 0.5) | 0.1918 | 0.9589 | 0.7884 | 0.9589 |

L'augmentation du poids BM25 dégrade systématiquement les scores sur ce dataset. 
Explication: les questions étant générées à partir du texte exact des chunks sources, elles favorisent structurellement la recherche sémantique; BM25 introduit plutôt du bruit lexical. 
La configuration retenue pour ce projet est la recherche sémantique seule (ou un poids BM25 faible, ≤0.2).

### Partie Génération

Modèle "gemini-2.5-pro" 

Les données du dataset ont été générées par le modèle Mistral. Chaque (Question/Réponse/Contexte) à été soumis à une évalution via Gemini 2.5 Flash comme modèle de juge (LLM-as-a-judge). L'objectif était de mesurer la performance sur la Fidélité et la Pertinence des réponses.


### Analyse des résultats

#### Résumé global (30 questions)
Modif eval à lancer seulement sur pertinente.

L'affichage des résultats des scores pour chaque questions afin de savoir quels questions a échoué:


> Modèle d'évaluation : Gemini 2.5 Pro via DeepEval  
> Dataset : 30 questions

---

#### Interprétation des métriques

##### Faithfulness ()

Un score de ... signifie ....

##### Answer Relevancy ()

Le score de ... est ....

---


#### Faiblesses identifiées et pistes de correction

A relancer apres modif










