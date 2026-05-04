"""Retry missing GSM8K evaluations with lower parallelism."""
import json, sys, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

client = OpenAI(timeout=60.0, max_retries=2)

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluations"
PERT_DIR = Path(__file__).parent.parent / "data" / "perturbations"

GSM_SYSTEM = "You are an expert at solving math word problems. Solve step by step. End with '#### <number>' on the final line."


def eval_gsm(model, question, correct_final):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": GSM_SYSTEM}, {"role": "user", "content": question}],
            temperature=0,
            max_tokens=600,
        )
        text = resp.choices[0].message.content
        m = re.search(r"####\s*([-\d,.]+)", text)
        if m:
            pred = m.group(1).strip().replace(",", "")
        else:
            nums = re.findall(r"[-]?\d+\.?\d*", text)
            pred = nums[-1] if nums else None
        if pred is None:
            return 0
        try:
            return 1 if abs(float(pred) - float(str(correct_final).replace(",", ""))) < 1e-4 else 0
        except:
            return 0
    except Exception as e:
        return None


def main():
    with open(PERT_DIR / "gsm8k_perturbed.json") as f:
        items = json.load(f)
    try:
        with open(EVAL_DIR / "gsm8k_results.json") as f:
            results = json.load(f)
    except:
        results = {}

    all_models = ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-08-06', 'gpt-4.1-2025-04-14']

    for model in all_models:
        if model not in results:
            results[model] = {"orig": [None]*len(items), "pert": [None]*len(items)}

        for version, qkey, akey in [("orig", "orig_question", "orig_final"), ("pert", "pert_question", "pert_final")]:
            current = results[model].get(version, [None]*len(items))
            if len(current) < len(items):
                current = current + [None] * (len(items) - len(current))
            missing = [i for i, x in enumerate(current) if x is None]
            if not missing:
                continue
            print(f"{model} {version}: retry {len(missing)} items", flush=True)

            with ThreadPoolExecutor(max_workers=12) as ex:
                futures = {ex.submit(eval_gsm, model, items[i][qkey], items[i][akey]): i for i in missing}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        current[idx] = fut.result()
                    except:
                        current[idx] = None
                    done += 1
                    if done % 25 == 0:
                        print(f"  {done}/{len(missing)}", flush=True)
                        results[model][version] = current
                        with open(EVAL_DIR / "gsm8k_results.json", 'w') as f:
                            json.dump(results, f, indent=2)

            results[model][version] = current
            with open(EVAL_DIR / "gsm8k_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            got = [x for x in current if x is not None]
            acc = sum(got)/len(got) if got else 0
            print(f"  {model} {version}: {len(got)}/{len(items)} complete, acc={acc:.3f}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
