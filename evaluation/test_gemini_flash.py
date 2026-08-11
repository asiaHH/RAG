"""
Test isole : verifie juste que GeminiJudge (flash + thinking_budget=1024)
produit un JSON valide, sans passer par DeepEval ni par le pipeline RAG complet.
Cout : 1 seul appel API, quelques centimes maximum.

Usage : python test_gemini_flash.py
A lancer depuis la racine du projet (pour que l'import evaluation.config fonctionne).
"""
import json
import re
from config import GeminiJudge


def strip_markdown_fences(text: str) -> str:
    """Retire les ```json ... ``` autour d'une réponse LLM, comme parse_llm_qa_json le fait
    déjà pour la génération du dataset. response_mime_type='application/json' ne semble pas
    toujours empêcher ChatGoogleGenerativeAI d'ajouter ces fences."""
    clean = text.strip()
    clean = re.sub(r'^```json\s*', '', clean)
    clean = re.sub(r'^```\s*', '', clean)
    clean = re.sub(r'```\s*$', '', clean).strip()
    return clean

# Prompt qui ressemble a ce que DeepEval envoie en interne pour Faithfulness
# (extraction de claims au format JSON) - pas le vrai prompt DeepEval exact,
# mais un test representatif du meme type de contrainte de formatage.
test_prompt = """Based on the given text, breakdown and generate a list of atomic, verifiable claims, to further processing.

Text:
Le stage a duré 6 mois et portait sur l'IA générative et le RAG.

**
IMPORTANT: Please make sure to only return in JSON format, with the "claims" key mapping to a list of strings.
Example JSON:
{
    "claims": ["claim 1", "claim 2"]
}
**

JSON:
"""

def test_json_validity(n_attempts: int = 4):
    """Repete l'appel plusieurs fois pour verifier que le JSON est valide de facon consistante,
    pas juste une fois par chance (les LLM ne sont pas parfaitement deterministes)."""
    judge = GeminiJudge(model="gemini-2.5-flash")

    successes = 0
    for attempt in range(1, n_attempts + 1):
        print(f"\n--- Tentative {attempt}/{n_attempts} ---")
        raw_output = judge.generate(test_prompt)
        print("Sortie brute :", raw_output[:300])
        cleaned = strip_markdown_fences(raw_output)

        try:
            parsed = json.loads(cleaned)
            assert "claims" in parsed
            print("✅ JSON valide :", parsed)
            successes += 1
        except (json.JSONDecodeError, AssertionError) as e:
            print("❌ JSON invalide :", e)

    print(f"\n=== Résultat : {successes}/{n_attempts} JSON valides ===")
    if successes == n_attempts:
        print("Le fix semble stable — tu peux relancer une éval réduite (5-10 questions) en conditions réelles.")
    elif successes > 0:
        print("Résultat inconsistant — le fix aide mais ne résout pas tout. Envisage d'augmenter thinking_budget encore.")
    else:
        print("Le fix ne suffit pas sur ce test. Vérifie response_mime_type ou reste sur pro pour l'instant.")


if __name__ == "__main__":
    test_json_validity()