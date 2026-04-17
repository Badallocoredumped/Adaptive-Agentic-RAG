"""
evaluation/prepare_squad_dataset.py
Loads SQuAD v1.1 validation set, builds:
  - evaluation/squad_corpus.json     -> unique passage contexts (your "documents")
  - evaluation/squad_eval_set.json   -> 100 sampled QA pairs with ground truth

Run: python evaluation/prepare_squad_dataset.py
"""
import json, random
from pathlib import Path
from datasets import load_dataset

random.seed(42)
EVAL_DIR = Path("evaluation")
EVAL_DIR.mkdir(exist_ok=True)
N_QUESTIONS = 100   # how many QA pairs to evaluate on

# -- Load SQuAD v1.1 validation split ----------------------------------------
print("Loading SQuAD v1.1...")
squad = load_dataset("rajpurkar/squad", split="validation")
print(f"Total validation examples: {len(squad)}")

# -- Build corpus: unique contexts keyed by a stable ID ----------------------
corpus = {}  # context_text -> {id, text, title}
for ex in squad:
    if ex["context"] not in corpus:
        corpus[ex["context"]] = {
            "context_id": f"ctx_{len(corpus):04d}",
            "text": ex["context"],
            "title": ex["title"],
        }

corpus_list = list(corpus.values())
print(f"Unique contexts (documents): {len(corpus_list)}")

with open(EVAL_DIR / "squad_corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus_list, f, indent=2)
print(f"Saved corpus -> evaluation/squad_corpus.json")

# -- Build eval set: sample N answerable questions ----------------------------
# Only keep questions where the answer is non-empty (answerable)
answerable = [ex for ex in squad if ex["answers"]["text"]]
sampled    = random.sample(answerable, N_QUESTIONS)

eval_set = []
for ex in sampled:
    ctx_id = corpus[ex["context"]]["context_id"]
    eval_set.append({
        "id":               ex["id"],
        "question":         ex["question"],
        "ground_truth":     ex["answers"]["text"][0],   # first human answer
        "all_answers":      ex["answers"]["text"],       # all valid answers
        "reference_context": ex["context"],             # the passage containing the answer
        "context_id":       ctx_id,
        "title":            ex["title"],
    })

with open(EVAL_DIR / "squad_eval_set.json", "w", encoding="utf-8") as f:
    json.dump(eval_set, f, indent=2)
print(f"Saved {len(eval_set)} QA pairs -> evaluation/squad_eval_set.json")
