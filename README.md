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

### Version actuelle (v1) — Retrieval sémantique seul
- [X] Évaluer le système v1 avec le dataset généré
- [X] Erreur retrouvé dans l'ingestion = doublons à corriger 
- [X] Ré-évaluer 

### Version 2 — Amélioration du retrieval
- [ ] Ajouter BM25 (recherche hybride sémantique + lexicale)
- [ ] Ré-évaluer et comparer avec v1

### Évaluation — Évolution prévue
Remplacement partiel de DeepEval pour limiter la consommation de quotas LLM :
- [X] Calcul des métriques de retrieval en Python pur (pas de LLM)
- [X] DeepEval conservé uniquement pour la génération du dataset
- [X] Gemini Flash remplacé par Gemini Pro car problème JSON rencontré 

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


### Stratégie d'évaluation : le dataset

Le dataset génératif sert à deux niveaux distincts : valider l'évaluateur lui-même, puis avoir confiance dans les résultats obtenus.

#### Niveau 1 — Valider l'évaluateur

Le dataset `generated_dataset_ratio_0.7.json` a été construit à partir du pipeline d'ingestion, sans annotation humaine. Le `ratio_0.7` signifie que 70% des questions ont leur `assigned_chunk_id` assigné de façon fiable.

#### Niveau 2 — Confiance dans les résultats

Une fois vérifié que l'évaluateur se comporte correctement sur ce dataset connu, les scores obtenus reflètent fidèlement les performances réelles du retriever.

## Rapport d'Évaluation de la Génération (V1)

### Partie Retrieval:

Evalué seulement sur les questions pertinentes du dataset soit 21 questions au lieu de 30. 
<img width="524" height="179" alt="Photos_2" src="https://github.com/user-attachments/assets/891dc824-41fe-4aa1-a5f3-905f895a5e95" />


Le retriever ne rate jamais le bon chunk dans les 5 résultats (Hit Rate 1.0), et il le met en première position ~87% du temps (MRR ~0.87). La Precision@K à 0.20 est une conséquence arithmétique du fait qu'on récupère 5 chunks pour 1 seul pertinent.

### Partie Génération

Problème avec le modèle "gemini-2.5-flash" : erreur structure JSON: 
>  Batch n échoué : Evaluation LLM outputted an invalid JSON. Please use a better evaluation model.


Modèle "gemini-2.5-pro": coût plus élévé
Lancement d'une evaluation génération et cout API assez élévé donc quelques modifications apportées: 
- Dataset de 30 questions au lieu de 50


Les données du dataset ont été générées par le modèle Mistral. Chaque (Question/Réponse/Contexte) à été soumis à une évalution via Gemini 2.5 Flash comme modèle de juge (LLM-as-a-judge). L'objectif était de mesurer la performance sur la Fidélité et la Pertinence des réponses.

*Objectif*: Utiliser le modèle Gemini pour détecter si Mistral a halluciné ou s'il a été imprécis.

### Analyse des résultats

#### Résumé global (30 questions)

<img width="446" height="84" alt="photos_3" src="https://github.com/user-attachments/assets/31537117-a865-4c3c-a7dc-fb3331fa287f" />

L'affichage des résultats des scores pour chaque questions afin de savoir quels questions a échoué:
<img width="497" height="527" alt="photos_1" src="https://github.com/user-attachments/assets/9bc44b9e-61e8-46a1-a687-a71c7215e212" />

> Modèle d'évaluation : Gemini 2.5 Pro via DeepEval  
> Dataset : 30 questions (pertinentes + hors-corpus)

---

#### Interprétation des métriques

##### Faithfulness (0.933)

Un score de 0.93 signifie que le RAG ne produit presque pas d'informations inventées ou déformées.

##### Answer Relevancy (0.849)

Le score de 0.85 est bon, mais c'est ici que se concentrent les principaux points d'amélioration.

---

#### Ce qui fonctionne bien

- **Fidélité au corpus élevée** : sur les questions dont la réponse est effectivement dans les documents, le modèle restitue correctement le contenu sans halluciner.
- **Pertinence des réponses techniques** : les questions précises sur du contenu bien délimité (MATLAB, méthode `paint`, etc.) obtiennent des scores parfaits (1.0/1.0 sur les deux métriques).
- **Pas de sur-invention** : le modèle ne "complète" pas le contexte avec des connaissances externes de façon erronée dans la grande majorité des cas.

---

#### Faiblesses identifiées et pistes de correction

##### 1. Questions hors-corpus — le RAG ne sait pas dire "je ne sais pas"

Quand la question n'a pas de réponse dans les documents indexés, le modèle produit malgré tout une réponse longue en s'accrochant à des chunks vaguement liés.

**Exemple observé** :

```
Question   : "Comment on inverse les fonctions ?"
Attendu    : "Information non disponible dans le corpus."
Obtenu     : Réponse de 600+ mots sur l'inversion mathématique, la factorielle JS,
             un arbre d'appels... basée sur des chunks hors-sujet récupérés par erreur.
Score AR   : 0.68 
```

Le retriever a remonté des chunks sur `f⁻¹` (image réciproque) et `fact()` (fonctions JS) parce que la question contient le mot "fonctions". Le LLM a synthétisé ces chunks de façon confiante au lieu de détecter l'absence de réponse réelle.

**Correction** — Améliorer le prompt système.

---

##### 2. Sur-interprétation sémantique — paraphrase infidèle

Le modèle reformule le contenu du chunk en changeant légèrement le sens, notamment sur des termes techniques précis.

**Exemple observé** :

```
Chunk source  : "limite géométrique : illusions optiques"
Réponse RAG   : "l'humain utilise des mécanismes cognitifs (mémoire, contexte,
                 illusions optiques) pour interpréter les images"
Chunk source  : "oeil travaille uniquement dans le visible"
Réponse RAG   : "les capteurs artificiels ont un spectre restreint" 
                (inversion du sens — c'est l'oeil qui est restreint, pas les capteurs)
Score Faith.  : 0.71 
```

Le modèle a inversé le sujet de la limitation spectrale et réinterprété "limite géométrique" en "mécanisme cognitif".

**Correction** — Améliorer le prompt.

---

##### 3. Réponses trop longues — dilution de la pertinence

Le modèle ajoute des informations exactes mais non demandées, ce qui dilue la réponse par rapport à la question.

**Exemple observé** :

```
Question  : "Quels sont les quatre points principaux de la conclusion ?"
Attendu   : Les 4 points listés précisément.
Obtenu    : 3 points listés + développement sur le "travail d'équipe"
            (information vraie du corpus, mais hors du champ de la question)
Score AR  : 0.71 
```

Le chunk sur le travail en équipe était adjacent dans le document au chunk de la conclusion. Le modèle l'a inclus parce qu'il était récupéré, sans vérifier s'il était demandé.

**Correction** — Améliorer le prompt.





