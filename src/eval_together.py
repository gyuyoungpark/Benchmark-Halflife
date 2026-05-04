"""
Together AI evaluation: Llama-3.3-70B-Instruct on GSM8K + MMLU-PRO perturbations.
Adds a 3rd vendor (open-source) to the cross-vendor contamination analysis.
"""
import json, os, sys, time
import requests
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_FILE = EVAL_DIR / "together_results.json"

API_KEY = os.environ.get("TOGETHER_API_KEY", "")
BASE_URL = "https://api.together.xyz/v1/chat/completions"

MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
]


def together_complete(model, prompt, max_tokens=10, temperature=0.0):
    for attempt in range(3):
        try:
            r = requests.post(BASE_URL,
                headers={"Authorization": f"Bearer {API_KEY}"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": max_tokens, "temperature": temperature},
                timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            elif r.status_code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                print(f"  ERR {r.status_code}: {r.text[:100]}", flush=True)
                return None
        except Exception as e:
            print(f"  ERR: {str(e)[:80]}", flush=True)
            time.sleep(2)
    return None


def eval_gsm8k(model):
    """Evaluate on 200 GSM8K items (orig + number-substituted)."""
    items = json.load(open(DATA_DIR / "perturbations" / "gsm8k_perturbed.json"))
    results = {"orig": [], "pert": []}

    print(f"\n=== GSM8K: {model} ({len(items)} items) ===", flush=True)
    for i, item in enumerate(items):
        for which in ["orig", "pert"]:
            q = item["orig_question"] if which == "orig" else item["pert_question"]
            if q is None:
                results[which].append(None)
                continue
            prompt = f"Solve this math problem step by step. At the end, write your final numerical answer after ####.\n\n{q}"
            resp = together_complete(model, prompt, max_tokens=500)
            if resp is None:
                results[which].append(None)
                continue
            # Extract answer after ####
            raw_ans = item["orig_answer"] if which == "orig" else item.get("pert_final", item.get("pert_answer", ""))
            # orig_answer is full solution text; extract final number after #### or last number
            if which == "orig" and "####" in str(raw_ans):
                correct_ans = str(raw_ans).split("####")[-1].strip()
            elif which == "orig":
                # Extract last number from solution text
                import re as _re
                _nums = _re.findall(r'-?\d[\d,]*\.?\d*', str(raw_ans))
                correct_ans = _nums[-1].replace(",", "") if _nums else str(raw_ans)
            else:
                correct_ans = raw_ans
            pred = ""
            if "####" in resp:
                pred = resp.split("####")[-1].strip()
            elif resp.strip():
                # Try last number
                import re
                nums = re.findall(r'-?\d+\.?\d*', resp)
                pred = nums[-1] if nums else ""
            # Normalize comparison
            try:
                correct = float(str(correct_ans).replace(",", "").strip())
                predicted = float(pred.replace(",", "").strip())
                results[which].append(1 if abs(correct - predicted) < 0.01 else 0)
            except:
                results[which].append(0)
            time.sleep(0.2)

        if (i + 1) % 25 == 0:
            o = [x for x in results["orig"] if x is not None]
            p = [x for x in results["pert"] if x is not None]
            o_acc = sum(o) / max(1, len(o))
            p_acc = sum(p) / max(1, len(p))
            print(f"  {i+1}/{len(items)}: orig={o_acc:.3f}, pert={p_acc:.3f}, gap={o_acc-p_acc:+.3f}", flush=True)

    return results


def eval_mmlu_pro(model):
    """Evaluate on MMLU-PRO items (orig + paraphrased)."""
    items = json.load(open(DATA_DIR / "perturbations" / "mmlu_pro_perturbed.json"))
    results = {"orig": [], "pert": []}

    print(f"\n=== MMLU-PRO: {model} ({len(items)} items) ===", flush=True)
    for i, item in enumerate(items):
        for which in ["orig", "pert"]:
            if which == "orig":
                q = item["orig_question"]
                choices = item["orig_choices"]
            else:
                q = item.get("pert_question", item["orig_question"])
                choices = item.get("pert_choices", item["orig_choices"])

            if q is None or choices is None:
                results[which].append(None)
                continue

            letters = "ABCDEFGHIJKLMNOP"
            choice_text = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
            prompt = f"Answer the following multiple-choice question. Reply with ONLY the letter (A, B, C, etc.).\n\n{q}\n\n{choice_text}\n\nAnswer:"

            resp = together_complete(model, prompt, max_tokens=5)
            if resp is None:
                results[which].append(None)
                continue

            # Extract letter
            pred_letter = ""
            for ch in resp.strip().upper():
                if ch in letters:
                    pred_letter = ch
                    break

            correct_letter = item.get("answer_letter", letters[item.get("answer_index", 0)])
            results[which].append(1 if pred_letter == correct_letter else 0)
            time.sleep(0.2)

        if (i + 1) % 25 == 0:
            o = [x for x in results["orig"] if x is not None]
            p = [x for x in results["pert"] if x is not None]
            o_acc = sum(o) / max(1, len(o))
            p_acc = sum(p) / max(1, len(p))
            print(f"  {i+1}/{len(items)}: orig={o_acc:.3f}, pert={p_acc:.3f}, gap={o_acc-p_acc:+.3f}", flush=True)

    return results


def main():
    existing = {}
    if OUT_FILE.exists():
        existing = json.load(open(OUT_FILE))

    for model in MODELS:
        model_key = model.split("/")[-1]

        # GSM8K
        gsm_key = f"{model_key}_gsm8k"
        if gsm_key not in existing:
            results = eval_gsm8k(model)
            existing[gsm_key] = results
            with open(OUT_FILE, "w") as f:
                json.dump(existing, f, indent=2)
            o = [x for x in results["orig"] if x is not None]
            p = [x for x in results["pert"] if x is not None]
            print(f"\nGSM8K DONE: orig={sum(o)/len(o):.3f}, pert={sum(p)/len(p):.3f}, gap={sum(o)/len(o)-sum(p)/len(p):+.3f}")
        else:
            print(f"Skip {gsm_key} (done)")

        # MMLU-PRO
        mmlu_key = f"{model_key}_mmlu_pro"
        if mmlu_key not in existing:
            results = eval_mmlu_pro(model)
            existing[mmlu_key] = results
            with open(OUT_FILE, "w") as f:
                json.dump(existing, f, indent=2)
            o = [x for x in results["orig"] if x is not None]
            p = [x for x in results["pert"] if x is not None]
            print(f"\nMMLU-PRO DONE: orig={sum(o)/len(o):.3f}, pert={sum(p)/len(p):.3f}, gap={sum(o)/len(o)-sum(p)/len(p):+.3f}")
        else:
            print(f"Skip {mmlu_key} (done)")


if __name__ == "__main__":
    main()
