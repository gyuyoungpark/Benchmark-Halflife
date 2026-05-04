"""
Generate perturbations for v2 benchmarks: MMLU-PRO (MCQ paraphrase) + MATH Lvl5 (number substitution).
Uses Claude haiku for generation.
"""
import json, os, sys, re, time
from pathlib import Path
from anthropic import Anthropic

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
ITEMS_DIR = DATA_DIR / "items"
PERT_DIR = DATA_DIR / "perturbations"
PERT_DIR.mkdir(parents=True, exist_ok=True)

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def paraphrase_mcq(question, choices, answer_idx):
    correct = LETTERS[answer_idx]
    choices_str = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    prompt = f"""Paraphrase this multiple-choice question while preserving meaning, difficulty, and the correct answer.

Rules:
1. Paraphrase the question stem with different wording, same meaning.
2. Keep all answer choices semantically identical (light rewording allowed).
3. The correct answer letter must remain {correct}.
4. Do NOT change technical terms, numbers, or named entities.
5. Output ONLY a JSON object with keys "question" (string) and "options" (array of {len(choices)} strings).

Original question:
{question}

Choices:
{choices_str}
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m: return None
        parsed = json.loads(m.group())
        if "question" in parsed and "options" in parsed and len(parsed["options"]) == len(choices):
            return parsed
    except Exception as e:
        print(f"    ERR: {str(e)[:80]}", flush=True)
    return None


def substitute_math(question, solution):
    prompt = f"""Rewrite this math problem with substituted numbers and names, preserving the EXACT same solution structure and operation order.

Rules:
1. Replace proper nouns and numbers; keep the operations and step count identical.
2. The new problem should have the same difficulty.
3. Output ONLY a JSON object with keys "question" (string), "solution" (worked solution), and "final_answer" (just the number or expression).

Original problem:
{question}

Original solution:
{solution}
"""
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5", max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        m = re.search(r'\{[\s\S]*\}', text)
        if not m: return None
        parsed = json.loads(m.group())
        if "question" in parsed and "final_answer" in parsed:
            return parsed
    except Exception as e:
        print(f"    ERR: {str(e)[:80]}", flush=True)
    return None


def generate_mmlu_pro():
    print("=== MMLU-PRO perturbations ===", flush=True)
    items = json.load(open(ITEMS_DIR / "mmlu_pro_items.json"))
    results = []
    for i, item in enumerate(items):
        pert = paraphrase_mcq(item['question'], item['options'], item['answer_index'])
        results.append({
            'orig_question': item['question'],
            'orig_choices': item['options'],
            'answer': item['answer_index'],
            'answer_letter': item['answer_letter'],
            'category': item['category'],
            'pert_question': pert['question'] if pert else None,
            'pert_choices': pert['options'] if pert else None,
        })
        if (i + 1) % 20 == 0:
            ok = sum(1 for r in results if r['pert_question'] is not None)
            print(f"  {i+1}/{len(items)} ({ok} ok)", flush=True)
        time.sleep(0.5)

    with open(PERT_DIR / "mmlu_pro_perturbed.json", 'w') as f:
        json.dump(results, f, indent=2)
    ok = sum(1 for r in results if r['pert_question'] is not None)
    print(f"  Done: {ok}/{len(results)} successful", flush=True)


def generate_math():
    print("=== MATH Lvl5 perturbations ===", flush=True)
    items = json.load(open(ITEMS_DIR / "math_lvl5_items.json"))
    results = []
    for i, item in enumerate(items):
        pert = substitute_math(item['question'], item['answer'])
        results.append({
            'orig_question': item['question'],
            'orig_answer': item['answer'],
            'topic': item['topic'],
            'pert_question': pert['question'] if pert else None,
            'pert_answer': pert.get('solution', '') if pert else None,
            'pert_final': str(pert.get('final_answer', '')) if pert else None,
        })
        if (i + 1) % 20 == 0:
            ok = sum(1 for r in results if r['pert_question'] is not None)
            print(f"  {i+1}/{len(items)} ({ok} ok)", flush=True)
        time.sleep(0.5)

    with open(PERT_DIR / "math_perturbed.json", 'w') as f:
        json.dump(results, f, indent=2)
    ok = sum(1 for r in results if r['pert_question'] is not None)
    print(f"  Done: {ok}/{len(results)} successful", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("mmlu_pro", "all"):
        generate_mmlu_pro()
    if cmd in ("math", "all"):
        generate_math()
