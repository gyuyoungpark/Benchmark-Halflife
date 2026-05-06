"""
Tier 1B: Arithmetic complexity matching for GSM8K 125-verified subset.

For each (orig, pert) pair, extract numeric inputs from the question and answer chain,
then compute four complexity features and check parity:
  - max digit length (in operands)
  - sum of digit lengths
  - carry/borrow count (approx, in addition/subtraction steps)
  - number of operations (already enforced as parity)
  - final-answer digit count

Report % of pairs that match per feature, plus a "all-match" rate.
"""
import json
import re
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"


def extract_numbers_from_question(q):
    """Extract integer/decimal numbers as floats."""
    nums = re.findall(r"-?\d+\.?\d*", q)
    return [float(n) for n in nums if n and n != "-"]


def extract_op_operands(answer_chain):
    """From <<a OP b = c>> markers, return list of (a, op, b)."""
    pat = re.compile(r"<<([\d\.]+)\s*([+\-*/])\s*([\d\.]+)\s*=\s*([\d\.]+)>>")
    return [(float(m.group(1)), m.group(2), float(m.group(3)), float(m.group(4)))
            for m in pat.finditer(answer_chain)]


def digit_count(x):
    if x == int(x):
        return len(str(int(abs(x))))
    return len(str(abs(x)).replace(".", "").lstrip("0")) or 1


def carries_in_add(a, b):
    """Count carry operations in adding two non-negative integers."""
    a, b = int(abs(a)), int(abs(b))
    c = 0
    carry = 0
    while a or b:
        s = a % 10 + b % 10 + carry
        if s >= 10:
            c += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return c


def features(item):
    q_orig = item["orig_question"]
    q_pert = item["pert_question"]
    a_orig = item["orig_answer"]
    a_pert = item["pert_answer"]
    f_orig = item["orig_final"]
    f_pert = item["pert_final"]

    # Numbers in question
    nq_o = extract_numbers_from_question(q_orig)
    nq_p = extract_numbers_from_question(q_pert)

    ops_o = extract_op_operands(a_orig)
    ops_p = extract_op_operands(a_pert)

    feat = {}
    feat["n_question_numbers"] = (len(nq_o), len(nq_p))
    feat["max_question_digits"] = (
        max((digit_count(n) for n in nq_o), default=0),
        max((digit_count(n) for n in nq_p), default=0),
    )
    feat["sum_question_digits"] = (
        sum(digit_count(n) for n in nq_o),
        sum(digit_count(n) for n in nq_p),
    )
    feat["n_ops"] = (len(ops_o), len(ops_p))
    # Carry count summed over additive ops
    co = sum(carries_in_add(a, b) for a, op, b, _ in ops_o if op in "+-")
    cp = sum(carries_in_add(a, b) for a, op, b, _ in ops_p if op in "+-")
    feat["carry_count"] = (co, cp)
    fo = int(str(f_orig).replace(",", ""))
    fp = int(str(f_pert).replace(",", ""))
    feat["final_digits"] = (digit_count(fo), digit_count(fp))
    return feat


def main():
    info = json.load(open(DATA / "gsm8k_verified" / "verified_subset.json"))
    pert = json.load(open(DATA / "perturbations" / "gsm8k_perturbed.json"))
    verified_idx = info["verified_idx"]
    n = len(verified_idx)
    print(f"Analyzing {n} verified GSM8K items.")

    feats = [features(pert[i]) for i in verified_idx]

    # Match rate per feature: pair matches if orig == pert exactly
    keys = ["n_question_numbers", "max_question_digits", "sum_question_digits",
            "n_ops", "carry_count", "final_digits"]
    print(f"\n{'Feature':25s}  exact-match%  diff-mean  diff-std")
    matches = {}
    for k in keys:
        diffs = np.array([f[k][1] - f[k][0] for f in feats])
        n_match = int((diffs == 0).sum())
        matches[k] = n_match / n
        print(f"  {k:25s}  {n_match}/{n} ({n_match/n*100:5.1f}%)   "
              f"{diffs.mean():+5.2f}     {diffs.std(ddof=1):.2f}")

    # All-match: every feature matches simultaneously
    all_match = sum(
        1 for f in feats if all(f[k][0] == f[k][1] for k in keys)
    )
    print(f"\nAll-{len(keys)}-features-match: {all_match}/{n} ({all_match/n*100:.1f}%)")

    # Strict subset: all features match
    strict_idx = [
        verified_idx[i] for i, f in enumerate(feats)
        if all(f[k][0] == f[k][1] for k in keys)
    ]
    print(f"Strict-equivalence subset size: {len(strict_idx)}")

    # Recompute orig-pert gap on strict subset
    results = json.load(open(DATA / "evaluations" / "gsm8k_results.json"))
    models = list(results.keys())
    o_mat = np.array([results[m]["orig"] for m in models])
    p_mat = np.array([results[m]["pert"] for m in models])
    if strict_idx:
        idx = np.array(strict_idx)
        gaps = (o_mat[:, idx] - p_mat[:, idx]).mean(axis=1)
        # item bootstrap
        item_avg = (o_mat[:, idx] - p_mat[:, idx]).mean(axis=0)
        rng = np.random.RandomState(123)
        boots = [item_avg[rng.randint(0, len(idx), size=len(idx))].mean()
                 for _ in range(5000)]
        ci = np.percentile(boots, [2.5, 97.5])
        print(f"\nStrict-subset gap (n={len(strict_idx)}): "
              f"{item_avg.mean()*100:+.2f}pp [95% CI {ci[0]*100:+.2f}, {ci[1]*100:+.2f}]")
        print(f"Per-model gaps:")
        for m, g in zip(models, gaps):
            print(f"  {m:30s} {g*100:+6.2f}pp")

    # Save
    out = {
        "n_verified": n,
        "match_rates": matches,
        "all_features_match_n": all_match,
        "all_features_match_frac": all_match / n,
        "strict_idx": strict_idx,
    }
    if strict_idx:
        out["strict_subset_gap_mean"] = float(item_avg.mean())
        out["strict_subset_gap_ci"] = [float(ci[0]), float(ci[1])]
    with open(DATA / "gsm8k_verified" / "arithmetic_complexity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {DATA / 'gsm8k_verified' / 'arithmetic_complexity.json'}")


if __name__ == "__main__":
    main()
