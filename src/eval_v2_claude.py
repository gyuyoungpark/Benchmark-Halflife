"""
Evaluate 3 Claude models on MMLU-PRO + MATH perturbation holdouts.
"""
import json, os, sys, re, time
from pathlib import Path
from anthropic import Anthropic

sys.stdout.reconfigure(line_buffering=True)
DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
MODELS = ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def eval_mc(model, question, choices, answer_idx):
    n = min(len(choices), 26)
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices[:n]))
    prompt = f"{question}\n\n{opts}\n\nOutput ONLY the letter of the correct answer."
    try:
        resp = client.messages.create(model=model, max_tokens=10,
            messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text.strip().upper()
        m = re.search(r"[A-Z]", text)
        if m:
            return 1 if m.group() == LETTERS[answer_idx] else 0
    except Exception as e:
        return None
    return 0


def eval_math(model, question):
    prompt = ("Solve this math problem step by step. "
              "End with '#### <answer>' on the final line.\n\n" + question)
    try:
        resp = client.messages.create(model=model, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}])
        text = resp.content[0].text
        m = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()
        nums = re.findall(r"[-]?\d+\.?\d*", text)
        return nums[-1] if nums else None
    except Exception:
        return None


def run_mmlu_pro():
    items = json.load(open(PERT_DIR / "mmlu_pro_perturbed.json"))
    items = [i for i in items if i.get('pert_question')]
    out_path = EVAL_DIR / "mmlu_pro_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    for model in MODELS:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skip {model} (done)", flush=True); continue
        print(f"\n=== MMLU-PRO {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        for label, qkey, ckey in [("orig", "orig_question", "orig_choices"),
                                   ("pert", "pert_question", "pert_choices")]:
            scores = []
            for i, it in enumerate(items):
                s = eval_mc(model, it[qkey], it[ckey], it['answer'])
                scores.append(s)
                if (i+1) % 40 == 0:
                    v = [x for x in scores if x is not None]
                    print(f"  [{label}] {i+1}/{len(items)} acc={sum(v)/max(1,len(v)):.3f}", flush=True)
                time.sleep(0.3)
            results[model][label] = scores
        json.dump(results, open(out_path, 'w'), indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        print(f"  {model}: orig={sum(o)/len(o):.3f} pert={sum(p)/len(p):.3f} gap={sum(o)/len(o)-sum(p)/len(p):+.3f}", flush=True)


def run_math():
    items = json.load(open(PERT_DIR / "math_perturbed.json"))
    items = [i for i in items if i.get('pert_question')]
    out_path = EVAL_DIR / "math_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    for model in MODELS:
        if model in results and len([x for x in results[model].get("orig", []) if x is not None]) > 100:
            print(f"Skip {model} (done)", flush=True); continue
        print(f"\n=== MATH {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        for label, qkey in [("orig", "orig_question"), ("pert", "pert_question")]:
            scores = []
            for i, it in enumerate(items):
                answer = eval_math(model, it[qkey])
                # Compare to original answer (extract final number)
                if label == "orig":
                    correct = re.findall(r"[-]?\d+\.?\d*", it.get('orig_answer', ''))
                    correct = correct[-1] if correct else None
                else:
                    correct = it.get('pert_final', '')
                if answer is None or correct is None:
                    scores.append(None)
                else:
                    try:
                        scores.append(1 if abs(float(answer) - float(str(correct).replace(',',''))) < 1e-2 else 0)
                    except:
                        scores.append(0)
                if (i+1) % 40 == 0:
                    v = [x for x in scores if x is not None]
                    print(f"  [{label}] {i+1}/{len(items)} acc={sum(v)/max(1,len(v)):.3f}", flush=True)
                time.sleep(0.5)
            results[model][label] = scores
        json.dump(results, open(out_path, 'w'), indent=2)
        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        if o and p:
            print(f"  {model}: orig={sum(o)/len(o):.3f} pert={sum(p)/len(p):.3f} gap={sum(o)/len(o)-sum(p)/len(p):+.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("mmlu_pro", "all"):
        run_mmlu_pro()
    if cmd in ("math", "all"):
        run_math()
    print("\nDone.", flush=True)
