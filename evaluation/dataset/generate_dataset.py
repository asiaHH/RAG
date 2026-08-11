import json
import random
import re
import time
import os
from dataclasses import asdict
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_not_exception_type
from evaluation.dataset.models import Chunk, QAPair, QuestionType, GOOD_DIMENSIONS, BAD_DIMENSIONS, DEFAULT_POSITIVE_RATIO, DEFAULT_N_QUESTIONS, DEFAULT_TOP_K_RETRIEVAL
from evaluation.dataset.clients import RealLLMClient, RealRetriever
from src.ingestion.loaders import clean_text

SLEEP_BETWEEN_CALLS = 1.0 #60 rpm default
CHECKPOINT_EVERY = 25  # save each N couples generated

def get_negative_chunk(chunks, chunk_i, chunk_split_ratios=(0.33, 0.66)):
    """
    Return a list of chunks for negative questions, prioritizing those less similar to the source chunk.
    Args:
        chunks: list of all available chunks
        chunk_i: id of the source chunk (the one used to generate the question)
        chunk_split_ratios: ratios for splitting chunks into tiers
    Returns:
        list: a list of candidate chunks for negative questions
    """
    n = len(chunks)
    tier1_end = int(n * chunk_split_ratios[0])
    tier2_end = int(n * chunk_split_ratios[1])
    first_tier = chunks[:tier1_end]
    third_tier = chunks[tier2_end:]
    
    source_index = next((i for i, c in enumerate(chunks) if c.id == chunk_i), None)
    if source_index is None:
        return [c for c in chunks if c.id != chunk_i]
    
    if source_index < tier1_end:
        pick_in = list(third_tier)
    elif source_index < tier2_end:
        pick_in = list(first_tier if random.random() < 0.5 else third_tier)
    else:
        pick_in = list(first_tier)
    
    return [c for c in pick_in if c.id != chunk_i]

def pick_negative_chunk(chunks, chunk_i):
    """
    Return a random chunk from the negative candidates for the given source chunk id.
    Args:
        chunks: list of all available chunks
        chunk_i: id of the source chunk (the one used to generate the question)
    Returns:
        Chunk or None: a random negative chunk, or None if no candidates are available
    """
    candidates = get_negative_chunk(chunks, chunk_i)
    return random.choice(candidates) if candidates else None

@retry(wait=wait_exponential(multiplier=2, max=60), stop=stop_after_attempt(6))
def round_trip_check(question, chunk_i, retriever, top_k=5):
    """
    Do a round-trip retrieval to check if the generated question can be retrieved from the source chunk.
    Args:
        question: the generated question to check
        chunk_i: id of the source chunk
        retriever: instance of RealRetriever to perform the search
        top_k: number of chunks to retrieve for the round-trip
    Returns:
        bool: True if the source chunk is among the top_k retrieved, False otherwise
    """
    retrieved = retriever.retrieve(question, top_k=top_k)
    retrieved_ids = {c.id for c in retrieved}

    return chunk_i in retrieved_ids

def build_generation_prompt(chunk_text, question_type, dimensions, is_positive):
    """
    Build a prompt for question generation, based on the question type, desired dimensions, and positive/negative character.
    Args:
        chunk_text: the text of the source chunk
        question_type: type of question to generate (QuestionType)
        dimensions: dict of dimensions to respect
        is_positive: bool indicating if the question should be relevant or not relevant
    Returns:
        str: the prompt to send to the LLM for question generation
    """
    dim_description = "\n".join(f"  - {dim}: {val}" for dim, val in dimensions.items())

    q_type_value= question_type.value if isinstance(question_type, QuestionType) else question_type
    type_instructions = {
        QuestionType.FACTUAL: "une question factuelle dont la réponse est explicitement présente dans le texte",
        QuestionType.REASONING: "une question de raisonnement dont la réponse se déduit logiquement du texte sans être explicite",
        QuestionType.SUMMARY: "une question de synthèse qui demande de résumer les points clés du texte",
        QuestionType.UNANSWERABLE: "une question dont la réponse N'EST PAS dans le texte (hors-corpus)",
        QuestionType.MULTI_HOP: "une question qui nécessite de combiner plusieurs informations du texte",
    }.get(q_type_value, "une question pertinente")
    
    relevance_note = "La question doit être directement répondable depuis ce texte." if is_positive else "La question doit sembler plausible mais NE PAS être répondable depuis ce texte."
    
    prompt = f"""Tu es un expert en création de datasets d'évaluation pour systèmes RAG.

TEXTE SOURCE :
\"\"\"
{chunk_text}
\"\"\"

CONSIGNE :
Génère {type_instructions}.
{relevance_note}

CARACTÉRISTIQUES DE LA QUESTION (respecte impérativement chaque dimension) :
{dim_description}

FORMAT DE RÉPONSE (JSON strict, aucun texte autour) :
{{
  "question": "<ta question>",
  "answer": "<réponse de référence, ou 'Information non disponible dans le corpus' si unanswerable>"
}}"""
    return prompt

def sample_dimensions(dim_config):
    """
    Choose a random value for each dimension from the provided options.
    Args:
        dim_config: dict where keys are dimension names and values are lists of possible values.
    Returns:
        dict: a dictionary containing the chosen values for each dimension.
    """
    return {dim: random.choice(values) for dim, values in dim_config.items()}

@retry(wait=wait_exponential(multiplier=2, max=60), stop=stop_after_attempt(6),retry=retry_if_not_exception_type(ValueError))
def parse_llm_qa_json(prompt, llm_client):
    """
    Send the prompt to the LLM and parse the JSON response to extract the question and answer.
    Args:
        prompt: the prompt to send to the LLM
        llm_client: instance of RealLLMClient for generating the question and answer
    Returns:
        tuple: (question, answer) extracted from the LLM's response
    """
    response = llm_client.generate(prompt)
    
    clean = response.strip()
    clean = re.sub(r'^```json\s*', '', clean)
    clean = re.sub(r'^```\s*', '', clean)
    clean = re.sub(r'```\s*$', '', clean).strip()
    
    # extract just the first valid JSON block
    match = re.search(r'\{.*?\}', clean, re.DOTALL)
    if not match:
        raise ValueError(f"Aucun JSON trouvé : {clean[:200]}")
        
    # correct the invalid backslashes (LaTeX formulas, etc.)
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match.group())
    
    data = json.loads(json_str)

    question=(data.get("question") or data.get("Question") or data.get("query"))
    answer=(data.get("answer") or data.get("réponse") or data.get("expected_answer"))

    if not question or not answer:
        raise ValueError(f"Question ou réponse manquante dans le JSON")

    return question, answer


def export_dataset(dataset, output_path):
    """
    Export the generated dataset to a JSON file, with a specific format.
    Args:
        dataset: list of QAPair objects to export
        output_path: path to the output JSON file
    """
    serialized = []
    for i, pair in enumerate(dataset, 1):
        item = asdict(pair)
        item['id'] = f"q{i}" 
        item['input'] = item.pop('question')
        item['expected_output'] = item.pop('answer')

        if hasattr(item['question_type'], 'value'):
            item['question_type'] = item['question_type'].value

        serialized.append(item)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"Dataset exporté → {output_path} ({len(dataset)} entrées)")
 
 
def load_checkpoint(checkpoint_path):
    """
    Reload a partial dataset from a checkpoint JSON file (same format as export_dataset).
    Args:
        checkpoint_path: path to the checkpoint JSON file
    Returns:
        list: list of QAPair objects rebuilt from the checkpoint, or [] if no checkpoint found
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return []
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = []
    skipped = 0
    for item in raw:
        if item.get("is_relevant", True) and "non disponible" in item.get("expected_output", "").lower():
            skipped += 1
            continue
        dataset.append(QAPair(
            question=item["input"],
            answer=item["expected_output"],
            source_chunk_id=item["source_chunk_id"],
            is_relevant=item["is_relevant"],
            assigned_chunk_id=item["assigned_chunk_id"],
            question_type=item["question_type"],
            dimensions=item["dimensions"],
            round_trip_passed=item.get("round_trip_passed"),
        ))
    return dataset


def generate_rag_dataset(chunks, llm_client, retriever=None, n_questions=DEFAULT_N_QUESTIONS, positive_ratio=DEFAULT_POSITIVE_RATIO, question_types=None, good_dimensions=None, bad_dimensions=None, use_round_trip=True, top_k=DEFAULT_TOP_K_RETRIEVAL, max_retries=3, checkpoint_path=None):
    """
    Generate a dataset of question-answer pairs for RAG evaluation, with a specified ratio of positive (relevant) and negative (non-relevant) questions.
    Args:
        chunks: list of Chunk objects to use as sources for question generation
        llm_client: instance of RealLLMClient for generating questions and answers
        retriever: instance of RealRetriever for round-trip checks (optional)
        n_questions: total number of questions to generate
        positive_ratio: desired ratio of positive (relevant) questions in the dataset
        question_types: list of QuestionType to choose from 
        good_dimensions: dict of dimensions for positive questions 
        bad_dimensions: dict of dimensions for negative questions 
        use_round_trip: whether to perform a round-trip retrieval check for positive questions
        top_k: number of chunks to retrieve for the round-trip check
        max_retries: number of attempts to generate a valid question-answer pair before giving up on a chunk
        checkpoint_path:if set, saves progress every CHECKPOINT_EVERY items and resumes from it if it already exists
    Returns:
        list: a list of QAPair objects representing the generated dataset
    """
    if question_types is None:
        question_types = [qt.value if hasattr(qt, 'value') else qt for qt in QuestionType]
    else:
        question_types = [qt.value if hasattr(qt, 'value') else qt for qt in question_types]

    unanswerable_value = QuestionType.UNANSWERABLE.value

    if good_dimensions is None:
        good_dimensions = GOOD_DIMENSIONS
    if bad_dimensions is None:
        bad_dimensions = BAD_DIMENSIONS
    
    dataset = load_checkpoint(checkpoint_path)
    generated_positive = sum(1 for pair in dataset if pair.is_relevant)
    generated_negative = len(dataset) - generated_positive
    if dataset:
        print(f"Reprise depuis le checkpoint : {len(dataset)} questions déjà générées ({generated_positive} positives, {generated_negative} négatives).")

    n_positive = int(n_questions * positive_ratio)
    n_negative = n_questions - n_positive
    
    print(f"Génération de {n_positive} questions positives et {n_negative} questions négatives...")
    
    while generated_positive + generated_negative < n_questions:
        still_need_positive = generated_positive < n_positive
        still_need_negative = generated_negative < n_negative
        
        if still_need_positive and still_need_negative:
            is_positive = random.random() < positive_ratio
        elif still_need_positive:
            is_positive = True
        else:
            is_positive = False
        
        source_chunk = random.choice(chunks)
        available_types = [t for t in question_types if t != unanswerable_value] if is_positive else question_types
        q_type = random.choice(available_types)
        dims = sample_dimensions(good_dimensions if is_positive else bad_dimensions)
        
        prompt = build_generation_prompt(source_chunk.text, q_type, dims, is_positive)
        
        question, answer = None, None
        for attempt in range(max_retries):
            try:
                question, answer = parse_llm_qa_json(prompt, llm_client)
                if is_positive and "non disponible" in answer.strip().lower():
                    raise ValueError("Réponse positive invalide")
                time.sleep(SLEEP_BETWEEN_CALLS)
                break
            except ValueError as e:
                print(f"  [Tentative {attempt + 1}/{max_retries}] Erreur génération : {e}")
        
        if question is None:
            print(f"  Abandon après {max_retries} tentatives pour chunk {source_chunk.id}")
            continue
        
        assigned_chunk = source_chunk if is_positive else pick_negative_chunk(chunks, source_chunk.id)
        if assigned_chunk is None:
            continue
        
        rtp = None
        if use_round_trip and retriever is not None:
            if is_positive and q_type != unanswerable_value:
                rtp = round_trip_check(question, source_chunk.id, retriever, top_k)
                time.sleep(SLEEP_BETWEEN_CALLS)
                if not rtp:
                    print(f"  Round-trip FAILED pour chunk {source_chunk.id} — question filtrée.")
                    continue
        
        pair = QAPair(
            question=question,
            answer=answer,
            source_chunk_id=source_chunk.id,
            is_relevant=is_positive,
            assigned_chunk_id=assigned_chunk.id,
            question_type=q_type,
            dimensions=dims,
            round_trip_passed=rtp,
        )
        dataset.append(pair)
        
        if is_positive:
            generated_positive += 1
        else:
            generated_negative += 1
        
        total = generated_positive + generated_negative
        if total % 10 == 0:
            print(f"  Progression : {total}/{n_questions} questions générées")

        if checkpoint_path and total % CHECKPOINT_EVERY == 0:
            export_dataset(dataset, checkpoint_path)
    
    print(f"\nDataset généré : {generated_positive} positives, {generated_negative} négatives.")
    return dataset


def generate_dataset_with_ratio(ratio: float, n_questions: int = 20, output_path: str = None):
    """
    Generate a RAG evaluation dataset with a specified ratio of positive (relevant) questions.
    Args:
        ratio: desired ratio of positive questions (e.g., 0.8 for 80%)
        n_questions: total number of questions to generate
    Returns:
        list: a list of QAPair objects representing the generated dataset
    """
    from src.ingestion.pipeline import init_vector_store
    vector_store = init_vector_store()
    
    all_docs = vector_store.similarity_search("", k=1000)
    chunks = [
    Chunk(
        id=doc.metadata.get("source_id", f"chunk_{i}"),
        text=clean_text(doc.page_content)
    )
    for i, doc in enumerate(all_docs)
]

    print(f"Chunks chargés depuis la base : {len(chunks)}")
    
    if output_path is None:
        output_path = f"evaluation/dataset/generated_dataset_V3_ratio_{ratio}.json"

    checkpoint_path= output_path.replace(".json", "_checkpoint.json")

    dataset = generate_rag_dataset(
        chunks=chunks,
        llm_client=RealLLMClient(),
        retriever=RealRetriever(vector_store),
        n_questions=n_questions,
        positive_ratio=ratio,
        use_round_trip=True,
        checkpoint_path=checkpoint_path
    )

    export_dataset(dataset, output_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"Dataset généré avec ratio {ratio} → {output_path}")
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Génère un dataset d'évaluation RAG")
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio de questions positives (ex. 0.8 pour 80%)")
    parser.add_argument("--n_questions", type=int, default=20, help="Nombre total de questions")
    parser.add_argument("--output", type=str, default=None, help="Chemin personnalisé du fichier JSON de sortie")
    args = parser.parse_args()
    
    generate_dataset_with_ratio(ratio=args.ratio, n_questions=args.n_questions, output_path=args.output)

