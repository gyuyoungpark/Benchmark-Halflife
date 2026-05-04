"""
Tier 2D: GSM8K human-verified equivalence subset.

Three-stage verification:
  Stage 1 (programmatic):
    - same operation count in <<...>> markers
    - pert has same final-answer type as orig (positive integer)
    - orig itself is unambiguous (>=6 of 7 models get it right)
    - perturbation is non-degenerate (pert_question differs from orig_question)
  Stage 2 (LLM-as-second-annotator, post-hoc): we leave a hook
    for an LLM annotator pass; for now we report the programmatic subset.
  Stage 3 (human spot-check): user manually verifies a random 10-item subset.

Output: data/gsm8k_verified/verified_subset.json,
        gap on the verified subset across 7 models.
"""
import json
import re
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"
OUT = DATA / "gsm8k_verified"
OUT.mkdir(parents=True, exist_ok=True)


def count_ops(s):
    return len(re.findall(r"<<.*?>>", s))


def final_answer(s):
    m = re.search(r"####\s*([\-0-9.]+)", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if v == int(v) else None
    except ValueError:
        return None


def is_int_positive(x):
    return x is not None and x == int(x) and x > 0


def main():
    pert = json.load(open(DATA / "perturbations" / "gsm8k_perturbed.json"))
    results = json.load(open(DATA / "evaluations" / "gsm8k_results.json"))

    # Per-item orig correctness across models: orig_correct[item_idx] = sum across models
    models = list(results.keys())
    n_items = len(pert)
    o_mat = np.array([results[m]["orig"] for m in models])  # (n_models, n_items)
    p_mat = np.array([results[m]["pert"] for m in models])

    verified_idx = []
    rejection_reasons = []
    for i, p in enumerate(pert):
        o_ops = count_ops(p["orig_answer"])
        p_ops = count_ops(p["pert_answer"])
        if o_ops != p_ops or o_ops < 1:
            rejection_reasons.append((i, "op_count_mismatch", o_ops, p_ops))
            continue
        o_final = final_answer(p["orig_answer"])
        p_final = final_answer(p["pert_answer"])
        if not (is_int_positive(o_final) and is_int_positive(p_final)):
            rejection_reasons.append((i, "non_int_or_negative_final", o_final, p_final))
            continue
        if o_final == p_final:
            rejection_reasons.append((i, "pert_same_final_as_orig", o_final, p_final))
            continue
        if p["orig_question"].strip() == p["pert_question"].strip():
            rejection_reasons.append((i, "pert_text_identical", None, None))
            continue
        n_orig_correct = int(o_mat[:, i].sum())
        if n_orig_correct < 6:
            rejection_reasons.append((i, "orig_ambiguous", n_orig_correct, None))
            continue
        verified_idx.append(i)

    print(f"Stage-1 programmatic verification: {len(verified_idx)}/{n_items} items pass")
    print(f"Rejections by reason:")
    from collections import Counter
    c = Counter(r[1] for r in rejection_reasons)
    for reason, n in c.most_common():
        print(f"  {reason}: {n}")

    # Random 50-item subset (seeded) for headline result; if <50 verified, take all
    rng = np.random.RandomState(42)
    if len(verified_idx) >= 50:
        subset_50 = sorted(rng.choice(verified_idx, size=50, replace=False).tolist())
    else:
        subset_50 = verified_idx
    print(f"\nHeadline subset size: {len(subset_50)}")

    # Compute gaps on verified subset
    sub_idx = np.array(subset_50)
    print(f"\nGap on verified subset ({len(sub_idx)} items):")
    print(f"{'Model':30s} {'orig':>6} {'pert':>6} {'gap':>7}")
    rows = []
    for i, m in enumerate(models):
        o = o_mat[i, sub_idx].mean()
        p = p_mat[i, sub_idx].mean()
        rows.append({"model": m, "orig": float(o), "pert": float(p), "gap": float(o - p)})
        print(f"{m:30s} {o:6.3f} {p:6.3f} {o-p:+7.3f}")

    # Bootstrap CI for mean gap across models
    gpt_models = [r for r in rows if r["model"].startswith("gpt")]
    claude_models = [r for r in rows if r["model"].startswith("claude")]
    gpt_gaps = np.array([r["gap"] for r in gpt_models])
    claude_gaps = np.array([r["gap"] for r in claude_models])
    print(f"\nGPT  mean gap = {gpt_gaps.mean():+.3f}  ({len(gpt_models)} models)")
    print(f"Claude mean gap = {claude_gaps.mean():+.3f}  ({len(claude_models)} models)")

    # Per-item bootstrap on the subset (model average per item)
    item_avg_gap = (o_mat[:, sub_idx] - p_mat[:, sub_idx]).mean(axis=0)
    rng2 = np.random.RandomState(123)
    boot_means = []
    for _ in range(2000):
        ix = rng2.randint(0, len(sub_idx), size=len(sub_idx))
        boot_means.append(float(item_avg_gap[ix].mean()))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"\nVerified-subset mean gap (item bootstrap, n={len(sub_idx)}): "
          f"{item_avg_gap.mean():+.3f} 95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    out = {
        "n_items_total": n_items,
        "n_items_verified": len(verified_idx),
        "verified_idx": verified_idx,
        "headline_subset_idx": subset_50,
        "headline_subset_size": len(subset_50),
        "rejection_breakdown": dict(c),
        "per_model_gap": rows,
        "gpt_mean_gap": float(gpt_gaps.mean()),
        "claude_mean_gap": float(claude_gaps.mean()),
        "subset_mean_gap": float(item_avg_gap.mean()),
        "subset_mean_gap_ci": [float(ci_lo), float(ci_hi)],
        "verification_criteria": [
            "operation_count_parity",
            "both_final_answers_positive_integers",
            "pert_final_differs_from_orig_final",
            "pert_text_differs_from_orig_text",
            "orig_unambiguous (>=6 of 7 models correct)",
        ],
    }
    with open(OUT / "verified_subset.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT / 'verified_subset.json'}")

    # Also dump the actual items for human spot-check
    spotcheck = [pert[i] for i in subset_50]
    with open(OUT / "verified_subset_items.json", "w") as f:
        json.dump(spotcheck, f, indent=2)
    print(f"Wrote items for spot-check: {OUT / 'verified_subset_items.json'}")


if __name__ == "__main__":
    main()
