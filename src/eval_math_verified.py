"""
Evaluate GPT and Claude models on the VERIFIED MATH perturbations (116 items).
Uses the same evaluation protocol as eval_v2_openai/claude but on the new verified set.
"""
import json, os, sys, re, time
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_FILE = EVAL_DIR / "math_verified_results.json"

MODELS_GPT = ["gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"]
MODELS_CLAUDE = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"]


def extract_answer(text):
    """Extract final answer from model response — robust nested brace parser."""
    if not text:
        return ""
    # Try #### first
    if "####" in text:
        return text.split("####")[-1].strip()
    # Try \\boxed{...} with nested braces
    idx = text.rfind("\\boxed{")
    if idx >= 0:
        depth = 0
        start = idx + 7
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    return text[start:j].strip()
                depth -= 1
    # Last line fallback
    lines = text.strip().split('\n')
    return lines[-1].strip()


def normalize_answer(s):
    """Normalize for comparison."""
    s = str(s).strip().lower()
    s = s.replace('$', '').replace('\\,', '').replace(' ', '')
    # Convert \frac{a}{b} to a/b for numeric comparison
    frac_match = re.match(r'\\?frac\{(\-?\d+)\}\{(\d+)\}', s)
    if frac_match:
        try:
            return float(frac_match.group(1)) / float(frac_match.group(2))
        except:
            pass
    try:
        return float(s.replace(',', ''))
    except:
        return s


def answers_match(a, b):
    na, nb = normalize_answer(a), normalize_answer(b)
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 0.01
    except:
        return str(na) == str(nb)


def openai_solve(model, question):
    from openai import OpenAI
    client = OpenAI(timeout=60.0, max_retries=2)
    prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}"
    try:
        r = client.chat.completions.create(
            model=model, max_tokens=1200, temperature=0.0,
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    except Exception as e:
        print(f"    GPT ERR: {str(e)[:80]}", flush=True)
        return None


def claude_solve(model, question):
    from anthropic import Anthropic
    client = Anthropic(timeout=60.0, max_retries=2)
    prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\n{question}"
    try:
        r = client.messages.create(
            model=model, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    except Exception as e:
        print(f"    Claude ERR: {str(e)[:80]}", flush=True)
        return None


def main():
    # Load verified perturbations (only verified=True items)
    all_items = json.load(open(DATA_DIR / "perturbations" / "math_verified_perturbed.json"))
    items = [it for it in all_items if it.get('verified', False)]
    print(f"=== MATH Verified Evaluation ({len(items)} verified items) ===", flush=True)

    existing = {}
    if OUT_FILE.exists():
        existing = json.load(open(OUT_FILE))

    # GPT models
    for model in MODELS_GPT:
        if model in existing and len([x for x in existing[model].get("orig", []) if x is not None]) > 50:
            print(f"Skip {model} (done)", flush=True)
            continue
        print(f"\n--- {model} ---", flush=True)
        results = {"orig": [], "pert": []}
        for i, item in enumerate(items):
            for which in ["orig", "pert"]:
                q = item["orig_question"] if which == "orig" else item["pert_question"]
                raw_correct = item["orig_answer"] if which == "orig" else item["pert_final"]
                correct = extract_answer(raw_correct)  # extract from solution text
                resp = openai_solve(model, q)
                if resp is None:
                    results[which].append(None)
                else:
                    pred = extract_answer(resp)
                    results[which].append(1 if answers_match(pred, correct) else 0)
                time.sleep(0.2)
            if (i + 1) % 20 == 0:
                o = [x for x in results["orig"] if x is not None]
                p = [x for x in results["pert"] if x is not None]
                print(f"  {i+1}/{len(items)}: orig={sum(o)/max(1,len(o)):.3f}, pert={sum(p)/max(1,len(p)):.3f}", flush=True)
        existing[model] = results
        with open(OUT_FILE, 'w') as f:
            json.dump(existing, f, indent=2)

    # Claude models
    for model in MODELS_CLAUDE:
        if model in existing and len([x for x in existing[model].get("orig", []) if x is not None]) > 50:
            print(f"Skip {model} (done)", flush=True)
            continue
        print(f"\n--- {model} ---", flush=True)
        results = {"orig": [], "pert": []}
        for i, item in enumerate(items):
            for which in ["orig", "pert"]:
                q = item["orig_question"] if which == "orig" else item["pert_question"]
                raw_correct = item["orig_answer"] if which == "orig" else item["pert_final"]
                correct = extract_answer(raw_correct)
                resp = claude_solve(model, q)
                if resp is None:
                    results[which].append(None)
                else:
                    pred = extract_answer(resp)
                    results[which].append(1 if answers_match(pred, correct) else 0)
                time.sleep(0.3)
            if (i + 1) % 20 == 0:
                o = [x for x in results["orig"] if x is not None]
                p = [x for x in results["pert"] if x is not None]
                print(f"  {i+1}/{len(items)}: orig={sum(o)/max(1,len(o)):.3f}, pert={sum(p)/max(1,len(p)):.3f}", flush=True)
        existing[model] = results
        with open(OUT_FILE, 'w') as f:
            json.dump(existing, f, indent=2)

    # Summary
    print("\n=== SUMMARY ===")
    from scipy.stats import pearsonr
    all_orig, all_pert = [], []
    for model in MODELS_GPT + MODELS_CLAUDE:
        if model not in existing:
            continue
        o = existing[model]["orig"]
        p = existing[model]["pert"]
        valid = [(ov, pv) for ov, pv in zip(o, p) if ov is not None and pv is not None]
        if not valid:
            continue
        ov, pv = zip(*valid)
        o_acc = np.mean(ov)
        p_acc = np.mean(pv)
        gap = o_acc - p_acc
        all_orig.extend(ov)
        all_pert.extend(pv)
        print(f"  {model}: orig={o_acc:.3f}, pert={p_acc:.3f}, gap={gap:+.3f} ({gap*100:+.1f}pp)")

    # Per-item fidelity
    n_items = len(items)
    n_models = len([m for m in MODELS_GPT + MODELS_CLAUDE if m in existing])
    item_orig = np.zeros(n_items)
    item_pert = np.zeros(n_items)
    item_count = np.zeros(n_items)
    for model in MODELS_GPT + MODELS_CLAUDE:
        if model not in existing:
            continue
        for i in range(min(n_items, len(existing[model].get("orig", [])))):
            o = existing[model]["orig"][i]
            p = existing[model]["pert"][i]
            if o is not None and p is not None:
                item_orig[i] += o
                item_pert[i] += p
                item_count[i] += 1
    valid_items = item_count > 0
    item_orig_acc = item_orig[valid_items] / item_count[valid_items]
    item_pert_acc = item_pert[valid_items] / item_count[valid_items]
    if len(item_orig_acc) > 3:
        r, p = pearsonr(item_orig_acc, item_pert_acc)
        print(f"\n  Per-item fidelity: r={r:.3f} (p={p:.4f}), n={len(item_orig_acc)} items")
        print(f"  Mean gap: {np.mean(item_orig_acc - item_pert_acc):+.4f} ({np.mean(item_orig_acc - item_pert_acc)*100:+.1f}pp)")


if __name__ == "__main__":
    main()
