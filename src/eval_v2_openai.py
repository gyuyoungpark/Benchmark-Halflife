"""Evaluate 4 GPT models on MMLU-PRO + MATH perturbation holdouts."""
import json, sys, re, time
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
DATA_DIR = Path(__file__).parent.parent / "data"
PERT_DIR = DATA_DIR / "perturbations"
EVAL_DIR = DATA_DIR / "evaluations"

client = OpenAI()
MODELS = ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def eval_mc(model, question, choices, answer_idx):
    n = min(len(choices), 26)
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices[:n]))
    try:
        resp = client.chat.completions.create(model=model, max_tokens=10,
            messages=[{"role":"user","content":f"{question}\n\n{opts}\n\nOutput ONLY the letter."}])
        text = resp.choices[0].message.content.strip().upper()
        m = re.search(r"[A-Z]", text)
        return 1 if m and m.group() == LETTERS[answer_idx] else 0
    except: return None

def eval_math(model, question):
    try:
        resp = client.chat.completions.create(model=model, max_tokens=1200,
            messages=[{"role":"user","content":"Solve step by step. End with '#### <answer>'.\n\n"+question}])
        text = resp.choices[0].message.content
        m = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
        return m.group(1).strip() if m else (re.findall(r"[-]?\d+\.?\d*", text) or [None])[-1]
    except: return None

def run_bench(bench, items, eval_fn, get_correct):
    out_path = EVAL_DIR / f"{bench}_results.json"
    results = json.load(open(out_path)) if out_path.exists() else {}
    for model in MODELS:
        if model in results and len([x for x in results[model].get("orig",[]) if x is not None]) > 50:
            print(f"Skip {model}", flush=True); continue
        print(f"\n=== {bench} {model} ===", flush=True)
        results[model] = {"orig": [], "pert": []}
        for label, qkey in [("orig","orig_question"),("pert","pert_question")]:
            scores = []
            for i, it in enumerate(items):
                if bench == "mmlu_pro":
                    s = eval_mc(model, it[qkey], it["orig_choices" if label=="orig" else "pert_choices"], it["answer"])
                else:
                    ans = eval_math(model, it[qkey])
                    correct = (re.findall(r"[-]?\d+\.?\d*", it.get("orig_answer","")) or [None])[-1] if label=="orig" else it.get("pert_final","")
                    try: s = 1 if ans and correct and abs(float(ans)-float(str(correct).replace(",","")))<1e-2 else 0
                    except: s = 0
                scores.append(s)
                if (i+1)%40==0:
                    v=[x for x in scores if x is not None]
                    print(f"  [{label}] {i+1}/{len(items)} acc={sum(v)/max(1,len(v)):.3f}", flush=True)
                time.sleep(0.2)
            results[model][label] = scores
        json.dump(results, open(out_path,'w'), indent=2)
        o=[x for x in results[model]["orig"] if x is not None]
        p=[x for x in results[model]["pert"] if x is not None]
        if o and p: print(f"  {model}: orig={sum(o)/len(o):.3f} pert={sum(p)/len(p):.3f} gap={sum(o)/len(o)-sum(p)/len(p):+.3f}", flush=True)

if __name__ == "__main__":
    mmlu_pro = [i for i in json.load(open(PERT_DIR/"mmlu_pro_perturbed.json")) if i.get("pert_question")]
    math = [i for i in json.load(open(PERT_DIR/"math_perturbed.json")) if i.get("pert_question")]
    cmd = sys.argv[1] if len(sys.argv)>1 else "all"
    if cmd in ("mmlu_pro","all"): run_bench("mmlu_pro", mmlu_pro, eval_mc, None)
    if cmd in ("math","all"): run_bench("math", math, eval_math, None)
    print("\nDone.", flush=True)
