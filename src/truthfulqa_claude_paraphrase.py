"""
Generate TruthfulQA perturbations via Claude (after OpenAI rate limit hit).
Re-paraphrases ALL 200 items via Claude haiku-4-5 for consistency.
"""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
from claude_eval import claude_paraphrase_mc

DATA_DIR = Path(__file__).parent.parent / "data"
ITEMS_PATH = DATA_DIR / "items" / "truthfulqa_items.json"
OUT_PATH = DATA_DIR / "perturbations" / "truthfulqa_perturbed.json"


def main():
    with open(ITEMS_PATH) as f:
        items = json.load(f)
    print(f"Paraphrasing {len(items)} TruthfulQA items via Claude...", flush=True)

    perturbed = []
    errors = 0
    done = 0

    def _one(item):
        mc1 = item.get("mc1_targets", {})
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices or not labels:
            return None
        try:
            correct_idx = labels.index(1)
        except ValueError:
            return None
        # Reorder so correct is at index 0
        ordered = [choices[correct_idx]] + [c for j, c in enumerate(choices) if j != correct_idx]
        # Now correct_idx is 0
        pert = claude_paraphrase_mc(item["question"], ordered, 0)
        if pert is None:
            return None
        return {
            "orig_question": item["question"],
            "orig_options": ordered,
            "pert_question": pert["question"],
            "pert_options": pert["options"],
            "correct_idx": 0,
        }

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_one, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r is None:
                errors += 1
            else:
                perturbed.append(r)
            if done % 25 == 0:
                print(f"  [{done}/{len(items)}] errors={errors}", flush=True)
                with open(OUT_PATH, 'w') as f:
                    json.dump(perturbed, f)

    with open(OUT_PATH, 'w') as f:
        json.dump(perturbed, f, indent=2)
    print(f"Done: {len(perturbed)}/{len(items)} ({errors} errors)", flush=True)


if __name__ == "__main__":
    main()
