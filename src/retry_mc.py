"""Retry missing MMLU/ARC pert evaluations."""
import json, sys, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
client = OpenAI(timeout=30.0, max_retries=2)

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluations"
PERT_DIR = Path(__file__).parent.parent / "data" / "perturbations"

MC_SYSTEM = """You are an expert at answering multiple-choice questions. For each question, output ONLY the letter of the correct answer (A, B, C, or D). Do not include explanations."""


def _clean_choice(c):
    if not isinstance(c, str):
        return str(c)
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s*", "", c).strip()


def eval_mc(model, question, choices, answer):
    letters = ["A", "B", "C", "D"]
    cleaned = [_clean_choice(c) for c in choices[:4]]
    prompt = f"{question}\n\n"
    for i, c in enumerate(cleaned):
        prompt += f"{letters[i]}) {c}\n"
    prompt += "\nAnswer:"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": prompt}],
            temperature=0, max_tokens=5)
        pred = resp.choices[0].message.content.strip().upper()
        m = re.search(r"[ABCD]", pred)
        if m:
            pred_letter = m.group()
            correct_letter = letters[answer] if isinstance(answer, int) else str(answer).upper()
            return 1 if pred_letter == correct_letter else 0
    except Exception:
        return None
    return 0


def main():
    for bench in ['mmlu', 'arc']:
        with open(PERT_DIR / f"{bench}_perturbed.json") as f:
            items = json.load(f)
        with open(EVAL_DIR / f"{bench}_results.json") as f:
            results = json.load(f)

        for model in ['gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06', 'gpt-4.1-2025-04-14']:
            if model not in results:
                continue
            current = results[model].get('pert', [])
            missing = [i for i, x in enumerate(current) if x is None]
            if not missing:
                continue
            print(f"{bench} {model} pert: retry {len(missing)} items", flush=True)

            with ThreadPoolExecutor(max_workers=12) as ex:
                if bench == 'mmlu':
                    futures = {ex.submit(eval_mc, model, items[i]['pert_question'],
                                         items[i]['pert_choices'], items[i]['answer']): i for i in missing}
                else:  # arc
                    futures = {ex.submit(eval_mc, model, items[i]['pert_question'],
                                         items[i]['pert_choices'], items[i]['answer_idx']): i for i in missing}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        current[idx] = fut.result()
                    except:
                        current[idx] = None
                    done += 1
                    if done % 30 == 0:
                        print(f"  {done}/{len(missing)}", flush=True)
                        results[model]['pert'] = current
                        with open(EVAL_DIR / f"{bench}_results.json", 'w') as f:
                            json.dump(results, f, indent=2)
            results[model]['pert'] = current
            with open(EVAL_DIR / f"{bench}_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            got = [x for x in current if x is not None]
            acc = sum(got)/len(got) if got else 0
            print(f"  Done: {len(got)}/{len(items)} valid, acc={acc:.3f}", flush=True)
    print("All done!", flush=True)


if __name__ == "__main__":
    main()
