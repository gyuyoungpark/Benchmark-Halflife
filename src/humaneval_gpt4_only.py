"""Retry HumanEval for gpt-4-turbo, gpt-4o, gpt-4.1 with lower parallelism."""
import json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from humaneval_pipeline import get_completion_openai, execute_solution, EVAL_PATH, PERT_PATH

sys.stdout.reconfigure(line_buffering=True)


def eval_one(model, item, version):
    prompt_key = f"{version}_prompt"
    entry_key = f"{version}_entry_point"
    test_key = f"{version}_test"
    completion = get_completion_openai(model, item[prompt_key])
    if completion is None:
        return None
    return execute_solution(completion, item[test_key], item[entry_key])


def main():
    with open(PERT_PATH) as f:
        items = json.load(f)
    with open(EVAL_PATH) as f:
        results = json.load(f)

    models = ["gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14"]

    for model in models:
        print(f"\n=== {model} ===", flush=True)
        if model not in results:
            results[model] = {"orig": [None]*len(items), "pert": [None]*len(items)}

        for version in ["orig", "pert"]:
            current = results[model].get(version, [None]*len(items))
            if len(current) < len(items):
                current = current + [None]*(len(items) - len(current))
            missing = [i for i, x in enumerate(current) if x is None]
            if not missing:
                print(f"  {version}: complete")
                continue
            print(f"  {version}: retry {len(missing)} items", flush=True)

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = {ex.submit(eval_one, model, items[i], version): i for i in missing}
                done = 0
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        current[idx] = fut.result()
                    except:
                        current[idx] = None
                    done += 1
                    if done % 20 == 0:
                        print(f"    {done}/{len(missing)}", flush=True)
                        results[model][version] = current
                        with open(EVAL_PATH, 'w') as f:
                            json.dump(results, f, indent=2)
            results[model][version] = current
            with open(EVAL_PATH, 'w') as f:
                json.dump(results, f, indent=2)

        o = [x for x in results[model]["orig"] if x is not None]
        p = [x for x in results[model]["pert"] if x is not None]
        oa = sum(o)/max(1, len(o)); pa = sum(p)/max(1, len(p))
        print(f"  {model}: orig={oa:.3f}({len(o)}/{len(items)}) pert={pa:.3f}({len(p)}/{len(items)}) gap={oa-pa:+.3f}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
