"""
Generate perturbation holdouts for MMLU, GSM8K, ARC-Challenge.

Strategy:
- MMLU/ARC (multiple choice): paraphrase the question stem + shuffle options
- GSM8K (math): substitute numbers while preserving solution structure

Each original item gets one perturbed version that is semantically equivalent
but surface-form distinct (to evade contamination).
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# Unbuffered stdout for progress visibility
sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data" / "items"
PERT_DIR = Path(__file__).parent.parent / "data" / "perturbations"
PERT_DIR.mkdir(parents=True, exist_ok=True)

client = OpenAI()

PARAPHRASE_MODEL = "gpt-4o-mini"  # cheap and fast


# ============================================================
# MMLU / ARC: multiple choice paraphrasing
# ============================================================

def build_mc_prompt(question, choices, answer_idx):
    letters = "ABCD"
    correct = letters[answer_idx] if isinstance(answer_idx, int) else str(answer_idx)
    choices_str = "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(choices[:4]))
    return f"""You will paraphrase a multiple-choice question while keeping its meaning, difficulty, and correct answer EXACTLY the same.

Rules:
1. Paraphrase the question stem using different wording but same meaning.
2. Keep all answer choices meaningfully identical (you may reword slightly).
3. The correct answer letter must remain the same.
4. Do NOT change any technical terms, numbers, or named entities.
5. Output ONLY a JSON object with two keys: "question" (string) and "choices" (array of 4 strings).

Original question:
{question}

Choices:
{choices_str}

Correct answer: {correct}
"""


def paraphrase_mc(question, choices, answer_idx):
    """Paraphrase a multiple-choice question."""
    prompt = build_mc_prompt(question, choices, answer_idx)

    try:
        resp = client.chat.completions.create(
            model=PARAPHRASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            "question": result["question"],
            "choices": result["choices"],
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# GSM8K: number substitution
# ============================================================

def build_gsm_prompt(question, answer):
    return f"""Rewrite this math word problem by substituting different numbers and names, but keep the EXACT same solution structure and reasoning steps. The final answer will be different but the problem should have the same difficulty.

Rules:
1. Replace proper nouns (names, locations) with different ones.
2. Replace numbers with different numbers that keep the problem solvable (avoid zeros, negatives unless present in original).
3. Keep the same operations in the same order.
4. Output ONLY a JSON object with three keys: "question" (string), "answer_text" (string, worked solution), and "final_answer" (string, just the number).

Original problem:
{question}

Original solution:
{answer}
"""


def paraphrase_gsm(question, answer):
    """Substitute numbers in GSM8K problem."""
    prompt = build_gsm_prompt(question, answer)

    try:
        resp = client.chat.completions.create(
            model=PARAPHRASE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content)
        return {
            "question": result["question"],
            "answer": result.get("answer_text", "") + f"\n#### {result.get('final_answer', '')}",
            "final_answer": result.get("final_answer", ""),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Pipeline
# ============================================================

def _mmlu_one(item):
    pert = paraphrase_mc(item["question"], item["choices"], item["answer"])
    if "error" in pert:
        return None
    return {
        "orig_question": item["question"],
        "orig_choices": item["choices"],
        "subject": item.get("subject", ""),
        "answer": item["answer"],
        "pert_question": pert["question"],
        "pert_choices": pert["choices"],
    }


def process_mmlu():
    print("Processing MMLU...", flush=True)
    with open(DATA_DIR / "mmlu_items.json") as f:
        items = json.load(f)
    out_path = PERT_DIR / "mmlu_perturbed.json"
    perturbed = []
    errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_mmlu_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is None:
                errors += 1
            else:
                perturbed.append(r)
            if done % 20 == 0:
                print(f"  [{done}/{len(items)}] errors={errors}", flush=True)
                with open(out_path, 'w') as f:
                    json.dump(perturbed, f)
    with open(out_path, 'w') as f:
        json.dump(perturbed, f, indent=2)
    print(f"  MMLU done: {len(perturbed)}/{len(items)} ({errors} errors)", flush=True)


def _arc_one(item):
    choices_texts = item["choices"]["text"] if isinstance(item["choices"], dict) else item["choices"]
    labels = item["choices"]["label"] if isinstance(item["choices"], dict) else ["A", "B", "C", "D"]
    answer_key = item["answerKey"]
    try:
        answer_idx = labels.index(answer_key)
    except ValueError:
        return None
    pert = paraphrase_mc(item["question"], choices_texts, answer_idx)
    if "error" in pert:
        return None
    return {
        "id": item["id"],
        "orig_question": item["question"],
        "orig_choices": choices_texts,
        "answer": answer_key,
        "answer_idx": answer_idx,
        "pert_question": pert["question"],
        "pert_choices": pert["choices"],
    }


def process_arc():
    print("Processing ARC-Challenge...", flush=True)
    with open(DATA_DIR / "arc_items.json") as f:
        items = json.load(f)
    out_path = PERT_DIR / "arc_perturbed.json"
    perturbed = []
    errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_arc_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is None:
                errors += 1
            else:
                perturbed.append(r)
            if done % 20 == 0:
                print(f"  [{done}/{len(items)}] errors={errors}", flush=True)
                with open(out_path, 'w') as f:
                    json.dump(perturbed, f)
    with open(out_path, 'w') as f:
        json.dump(perturbed, f, indent=2)
    print(f"  ARC done: {len(perturbed)}/{len(items)} ({errors} errors)", flush=True)


def _gsm_one(item):
    m = re.search(r"####\s*([-\d,.]+)", item["answer"])
    orig_final = m.group(1).strip() if m else None
    pert = paraphrase_gsm(item["question"], item["answer"])
    if "error" in pert:
        return None
    return {
        "orig_question": item["question"],
        "orig_answer": item["answer"],
        "orig_final": orig_final,
        "pert_question": pert["question"],
        "pert_answer": pert["answer"],
        "pert_final": pert.get("final_answer", ""),
    }


def process_gsm8k():
    print("Processing GSM8K...", flush=True)
    with open(DATA_DIR / "gsm8k_items.json") as f:
        items = json.load(f)
    out_path = PERT_DIR / "gsm8k_perturbed.json"
    perturbed = []
    errors = 0
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = {ex.submit(_gsm_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is None:
                errors += 1
            else:
                perturbed.append(r)
            if done % 20 == 0:
                print(f"  [{done}/{len(items)}] errors={errors}", flush=True)
                with open(out_path, 'w') as f:
                    json.dump(perturbed, f)
    with open(out_path, 'w') as f:
        json.dump(perturbed, f, indent=2)
    print(f"  GSM8K done: {len(perturbed)}/{len(items)} ({errors} errors)", flush=True)


if __name__ == "__main__":
    import sys
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "all"
    if benchmark in ("mmlu", "all"):
        process_mmlu()
    if benchmark in ("arc", "all"):
        process_arc()
    if benchmark in ("gsm8k", "all"):
        process_gsm8k()
    print("\nAll done!")
