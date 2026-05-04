"""Run GSM8K eval on specific models, merging with existing results."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_models import run_gsm, MODELS, EVAL_DIR

# Only run missing models
with open(EVAL_DIR / "gsm8k_results.json") as f:
    existing = json.load(f)

done_models = [m for m, v in existing.items() if len([x for x in v.get('orig', []) if x is not None]) > 100]
print(f"Already done: {done_models}", flush=True)

todo = [m for m in MODELS if m not in done_models]
print(f"Todo: {todo}", flush=True)

if todo:
    new_results = run_gsm(models=todo)
    # Merge
    with open(EVAL_DIR / "gsm8k_results.json") as f:
        merged = json.load(f)
    for m, v in new_results.items():
        merged[m] = v
    with open(EVAL_DIR / "gsm8k_results.json", 'w') as f:
        json.dump(merged, f, indent=2)

print("Done!", flush=True)
