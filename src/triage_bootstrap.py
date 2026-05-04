"""
W1: Triage simulation with bootstrap CIs.

Extends triage_simulation.py by:
  (a) Item-level bootstrap: resample perturbation items within each benchmark
      to propagate Monte-Carlo error from the oracle accuracy estimates.
  (b) Model-level bootstrap: resample the 7-model pool with replacement to
      quantify how much the reported Spearman rho depends on the specific
      models chosen.

Outputs rho ± 95% bootstrap CI per (policy, k), plus a head-to-head p-value
for halflife_long vs random.
"""
import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = DATA_DIR / "evaluations"
OUT_DIR = DATA_DIR / "triage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HALFLIFE = {
    "winogrande": 1.7, "hellaswag": 2.5, "arc": 3.4, "gsm8k": 4.9,
    "truthfulqa": 5.7, "bbh": 19.6,
    "mmlu": 50.0, "mmlu_pro": 60.0, "gpqa": 60.0, "math": 60.0,
    "ifeval": 60.0, "musr": 60.0,
}
AKHTAR_S = {
    "arc": 0.252, "hellaswag": 0.153, "truthfulqa": 0.003, "winogrande": 0.106,
    "gsm8k": 0.071, "bbh": 0.534, "mmlu_pro": 0.966, "gpqa": 0.784,
    "ifeval": 0.994, "math": 0.630, "musr": 0.416, "mmlu": 0.40,
}
MODELS = [
    "gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06",
    "gpt-4.1-2025-04-14", "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
]
BENCHES = ["mmlu", "arc", "gsm8k", "truthfulqa", "humaneval"]


def load_pert_items():
    """Return {bench: {model: np.array of 0/1 per-item pert accuracy}}."""
    out = {}
    for b in BENCHES:
        path = EVAL_DIR / f"{b}_results.json"
        if not path.exists():
            continue
        r = json.load(open(path))
        col = {}
        for m in MODELS:
            if m not in r:
                continue
            pert = [float(x) for x in r[m].get("pert", []) if x is not None]
            if len(pert) >= 50:
                col[m] = np.array(pert)
        if col:
            out[b] = col
    return out


def pick_benches(policy, k):
    benches = list(BENCHES)
    if policy == "random":
        rng = np.random.RandomState(0)
        return list(rng.choice(benches, size=min(k, len(benches)), replace=False))
    if policy == "akhtar_low":
        return sorted(benches, key=lambda b: AKHTAR_S.get(b, 1.0))[:k]
    if policy == "akhtar_high":
        return sorted(benches, key=lambda b: -AKHTAR_S.get(b, 0.0))[:k]
    if policy == "halflife_long":
        return sorted(benches, key=lambda b: -HALFLIFE.get(b, 60.0))[:k]
    if policy == "halflife_short":
        return sorted(benches, key=lambda b: HALFLIFE.get(b, 60.0))[:k]
    raise ValueError(policy)


def rho_for_draw(items, models, chosen, rng, item_bootstrap=True, model_bootstrap=False):
    """One bootstrap replicate of Spearman rho."""
    if model_bootstrap:
        idx = rng.randint(0, len(models), size=len(models))
        models = [models[i] for i in idx]

    def mean_over(bench_list):
        scores = {}
        for m in models:
            vals = []
            for b in bench_list:
                if b not in items or m not in items[b]:
                    continue
                arr = items[b][m]
                if item_bootstrap:
                    s = arr[rng.randint(0, len(arr), size=len(arr))].mean()
                else:
                    s = arr.mean()
                vals.append(s)
            if vals:
                scores[m] = np.mean(vals)
        return scores

    oracle = mean_over(list(items.keys()))
    policy = mean_over(chosen)
    common = [m for m in set(oracle) & set(policy)]
    if len(common) < 3:
        return np.nan
    o_rank = np.argsort(np.argsort([-oracle[m] for m in common]))
    p_rank = np.argsort(np.argsort([-policy[m] for m in common]))
    rho, _ = spearmanr(o_rank, p_rank)
    return rho


def bootstrap_policy(items, policy, k, n_boot=2000, seed=42, model_boot=False):
    chosen = pick_benches(policy, k)
    rng = np.random.RandomState(seed)
    rhos = np.array([rho_for_draw(items, MODELS, chosen, rng,
                                  item_bootstrap=True,
                                  model_bootstrap=model_boot)
                     for _ in range(n_boot)])
    rhos = rhos[~np.isnan(rhos)]
    return {
        "chosen": chosen,
        "rho_mean": float(np.mean(rhos)),
        "rho_median": float(np.median(rhos)),
        "ci_lo": float(np.percentile(rhos, 2.5)),
        "ci_hi": float(np.percentile(rhos, 97.5)),
        "n_boot": len(rhos),
        "rhos": rhos,
    }


def head_to_head(items, policy_a, policy_b, k, n_boot=2000, seed=7, model_boot=False):
    """Paired bootstrap: P(rho_a > rho_b) under item+model resampling."""
    chosen_a = pick_benches(policy_a, k)
    chosen_b = pick_benches(policy_b, k)
    rng = np.random.RandomState(seed)
    wins = 0
    diffs = []
    for _ in range(n_boot):
        # Use same bootstrap draw for both policies (paired)
        state = rng.get_state()
        ra = rho_for_draw(items, MODELS, chosen_a, rng, True, model_boot)
        rng.set_state(state)
        rb = rho_for_draw(items, MODELS, chosen_b, rng, True, model_boot)
        if np.isnan(ra) or np.isnan(rb):
            continue
        diffs.append(ra - rb)
        if ra > rb:
            wins += 1
    diffs = np.array(diffs)
    return {
        "p_a_beats_b": wins / len(diffs),
        "mean_diff": float(np.mean(diffs)),
        "ci_diff": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
    }


def main():
    items = load_pert_items()
    print(f"Loaded {len(items)} benchmarks with per-item pert accuracy")
    for b, col in items.items():
        print(f"  {b}: {len(col)} models, {list(col.values())[0].shape[0]} items")

    policies = ["random", "akhtar_low", "akhtar_high", "halflife_long", "halflife_short"]

    all_results = {"item_only": {}, "item_and_model": {}}
    for model_boot in [False, True]:
        key = "item_and_model" if model_boot else "item_only"
        print(f"\n{'='*60}\nBootstrap: {key}\n{'='*60}")
        for k in [1, 2, 3]:
            all_results[key][k] = {}
            for p in policies:
                r = bootstrap_policy(items, p, k, n_boot=2000, model_boot=model_boot)
                all_results[key][k][p] = {kk: v for kk, v in r.items() if kk != "rhos"}
                print(f"  k={k} {p:<18} rho={r['rho_mean']:+.3f} "
                      f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  chosen={r['chosen']}")

    # Head-to-head: halflife_long vs random at k=3
    print(f"\n{'='*60}\nHead-to-head: halflife_long vs random (k=3)\n{'='*60}")
    for model_boot in [False, True]:
        key = "item_and_model" if model_boot else "item_only"
        h2h = head_to_head(items, "halflife_long", "random", k=3,
                           n_boot=2000, model_boot=model_boot)
        print(f"  {key}: P(hl_long > random) = {h2h['p_a_beats_b']:.3f}, "
              f"mean Δρ = {h2h['mean_diff']:+.3f} [{h2h['ci_diff'][0]:+.3f}, {h2h['ci_diff'][1]:+.3f}]")
        all_results.setdefault("head_to_head", {})[key] = h2h

    with open(OUT_DIR / "triage_bootstrap.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'triage_bootstrap.json'}")


if __name__ == "__main__":
    main()
