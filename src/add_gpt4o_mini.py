"""Add gpt-4o-mini to MMLU/ARC/GSM8K/TruthfulQA evaluations."""
import json, sys, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
client = OpenAI(timeout=30.0, max_retries=2)

DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"
MODEL = "gpt-4o-mini-2024-07-18"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _clean_choice(c):
    if not isinstance(c, str): return str(c)
    return re.sub(r"^\s*[A-Da-d][\)\.\:]\s*", "", c).strip()


def eval_mc(question, choices, answer):
    n = min(len(choices), 26)
    letters = list(LETTERS[:n])
    cleaned = [_clean_choice(c) for c in choices[:n]]
    prompt = f"{question}\n\n" + "\n".join(f"{letters[i]}) {c}" for i, c in enumerate(cleaned)) + "\n\nOutput ONLY the letter."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=5)
        pred = resp.choices[0].message.content.strip().upper()
        m = re.search(r"[A-Z]", pred)
        if m:
            try:
                return 1 if letters.index(m.group()) == (answer if isinstance(answer, int) else letters.index(answer)) else 0
            except: return 0
    except: return None
    return 0


def eval_gsm(question, correct):
    prompt = "Solve step by step. End with '#### <number>'.\n\n" + question
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=600)
        text = resp.choices[0].message.content
        m = re.search(r"####\s*([-\d,.]+)", text)
        pred = m.group(1).strip().replace(",", "") if m else None
        if pred is None:
            nums = re.findall(r"[-]?\d+\.?\d*", text)
            pred = nums[-1] if nums else None
        if pred is None: return 0
        try:
            return 1 if abs(float(pred) - float(str(correct).replace(",", ""))) < 1e-4 else 0
        except: return 0
    except: return None


def eval_tqa(question, options, correct_idx):
    n = min(len(options), 26)
    letters = list(LETTERS[:n])
    options_str = "\n".join(f"{letters[i]}) {o}" for i, o in enumerate(options[:n]))
    prompt = f"{question}\n\n{options_str}\n\nOutput ONLY the letter."
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=5)
        pred = resp.choices[0].message.content.strip().upper()
        m = re.search(r"[A-Z]", pred)
        if m:
            try: return 1 if letters.index(m.group()) == correct_idx else 0
            except: return 0
    except: return None
    return 0


def _eval_parallel(eval_fn, tasks, workers=12):
    results = [None] * len(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(eval_fn, *args): idx for idx, args in tasks}
        for fut in as_completed(futures):
            idx = futures[fut]
            try: results[idx] = fut.result()
            except: results[idx] = None
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(tasks)}", flush=True)
    return results


def run_one(name, eval_fn, items, key_fn):
    eval_path = EVAL_DIR / f"{name}_results.json"
    with open(eval_path) as f:
        results = json.load(f)
    if MODEL in results and len([x for x in results[MODEL].get("orig", []) if x is not None]) > 100:
        print(f"{name}: {MODEL} already done")
        return
    print(f"\n=== {name.upper()} on {MODEL} ===", flush=True)
    results[MODEL] = {"orig": [], "pert": []}
    for version_key in ["orig", "pert"]:
        print(f"  [{version_key}]", flush=True)
        tasks = [(i, key_fn(items[i], version_key)) for i in range(len(items))]
        results[MODEL][version_key] = _eval_parallel(eval_fn, tasks)
    with open(eval_path, 'w') as f:
        json.dump(results, f, indent=2)
    o = [x for x in results[MODEL]["orig"] if x is not None]
    p = [x for x in results[MODEL]["pert"] if x is not None]
    print(f"  orig={sum(o)/max(1,len(o)):.3f}({len(o)}) pert={sum(p)/max(1,len(p)):.3f}({len(p)})", flush=True)


def main():
    # MMLU
    with open(PERT_DIR / "mmlu_perturbed.json") as f:
        items = json.load(f)
    run_one("mmlu", eval_mc, items, lambda it, v: (it[f"{v}_question"], it[f"{v}_choices"], it["answer"]))

    # ARC
    with open(PERT_DIR / "arc_perturbed.json") as f:
        items = json.load(f)
    run_one("arc", eval_mc, items, lambda it, v: (it[f"{v}_question"], it[f"{v}_choices"], it["answer_idx"]))

    # GSM8K
    with open(PERT_DIR / "gsm8k_perturbed.json") as f:
        items = json.load(f)
    run_one("gsm8k", eval_gsm, items, lambda it, v: (it[f"{v}_question"], it[f"{v}_final"]))

    # TruthfulQA
    with open(PERT_DIR / "truthfulqa_perturbed.json") as f:
        items = json.load(f)
    run_one("truthfulqa", eval_tqa, items, lambda it, v: (it[f"{v}_question"], it[f"{v}_options"], it["correct_idx"]))

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
