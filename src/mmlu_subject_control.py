"""MMLU per-subject gap control for the MMLU-PRO subject-heterogeneity result.

The MMLU-PRO orig-pert gap is concentrated in humanities/social-science
subjects (history +15.5pp, economics +13.1pp, psychology +8.3pp; STEM ≈ 0).
Reviewer concern: this could be a paraphrase-difficulty artifact rather
than a contamination/memorisation signal.

Control test: MMLU has higher per-item fidelity (r = 0.88 vs MMLU-PRO 0.83)
on the same paraphrase pipeline. If the humanities-elevation pattern were
purely a paraphrase-style artifact, we'd expect MMLU humanities subjects
to show a comparable elevation. We compute per-subject and per-domain-cluster
gaps on MMLU and report the comparison.

We also test, on MMLU-PRO, whether per-item gap correlates with per-item
fidelity (orig-pert agreement across models). A strong negative correlation
would imply that high-gap items are simply low-fidelity items (artifact);
a near-zero or positive correlation rules that out.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# MMLU subject → high-level cluster
HUMANITIES = {
    "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "prehistory", "world_religions",
    "philosophy", "moral_disputes", "moral_scenarios",
    "logical_fallacies", "formal_logic",
    "international_law", "jurisprudence", "professional_law",
    "us_foreign_policy",
}
SOCIAL_SCIENCE = {
    "high_school_psychology", "professional_psychology",
    "sociology", "marketing", "public_relations",
    "high_school_macroeconomics", "high_school_microeconomics",
    "econometrics",
    "high_school_government_and_politics", "security_studies",
    "human_sexuality", "global_facts",
    "miscellaneous",
}
STEM = {
    "abstract_algebra", "college_mathematics", "elementary_mathematics",
    "high_school_mathematics", "high_school_statistics",
    "college_physics", "high_school_physics", "conceptual_physics",
    "astronomy", "college_chemistry", "high_school_chemistry",
    "college_biology", "high_school_biology", "anatomy",
    "college_computer_science", "high_school_computer_science",
    "computer_security", "machine_learning", "electrical_engineering",
    "college_medicine", "professional_medicine", "clinical_knowledge",
    "medical_genetics", "human_aging", "nutrition", "virology",
    "high_school_geography",
}
PROF_OTHER = {
    "business_ethics", "management", "professional_accounting",
}


def cluster_of(subject):
    if subject in HUMANITIES:
        return "Humanities"
    if subject in SOCIAL_SCIENCE:
        return "Social science"
    if subject in STEM:
        return "STEM"
    return "Other"


def main():
    # ---- Load MMLU items + per-model orig/pert outcomes ----
    items = json.load(open(DATA / "perturbations" / "mmlu_perturbed.json"))
    results = json.load(open(DATA / "evaluations" / "mmlu_results.json"))

    # Match MMLU-PRO analysis: use 6 frontier models
    frontier = [
        "gpt-4-turbo-2024-04-09", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14",
        "claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5",
    ]
    frontier = [m for m in frontier if m in results]
    print(f"Frontier models used: {len(frontier)}: {frontier}")
    n_items = len(items)
    print(f"MMLU items: {n_items}")

    # Per-item orig/pert mean accuracy across frontier models
    orig_acc = np.zeros(n_items)
    pert_acc = np.zeros(n_items)
    for m in frontier:
        orig_acc += np.array(results[m]["orig"], dtype=float)
        pert_acc += np.array(results[m]["pert"], dtype=float)
    orig_acc /= len(frontier)
    pert_acc /= len(frontier)
    gap = orig_acc - pert_acc

    # Per-subject and per-cluster gap
    by_cluster = defaultdict(list)
    by_subject = defaultdict(list)
    for i, it in enumerate(items):
        by_cluster[cluster_of(it["subject"])].append(gap[i])
        by_subject[it["subject"]].append(gap[i])

    print("\n=== MMLU gap by cluster (mean across", len(frontier), "frontier models) ===")
    for c in ["Humanities", "Social science", "STEM", "Other"]:
        if c in by_cluster:
            arr = np.array(by_cluster[c])
            print(f"  {c:15s}  n={len(arr):3d}  mean gap = {arr.mean()*100:+.1f}pp  ({arr.std(ddof=1)*100:.1f} sd)")

    # Top humanities-leaning subjects (sample)
    print("\n=== Top-5 MMLU humanities subjects by mean gap ===")
    hum_subj = [(s, np.mean(g)) for s, g in by_subject.items()
                if cluster_of(s) in {"Humanities", "Social science"}]
    for s, mg in sorted(hum_subj, key=lambda x: -x[1])[:5]:
        n = len(by_subject[s])
        print(f"  {s:38s}  n={n:2d}  mean gap = {mg*100:+.1f}pp")

    # ---- MMLU-PRO: per-item gap vs per-item fidelity correlation ----
    pro_results = json.load(open(DATA / "evaluations" / "mmlu_pro_results.json"))
    pro_frontier = [m for m in frontier if m in pro_results]
    n_pro = len(pro_results[pro_frontier[0]]["orig"])
    print(f"\nMMLU-PRO items: {n_pro}; frontier models: {len(pro_frontier)}")

    # Build (n_models, n_items) matrices
    O = np.array([[x for x in pro_results[m]["orig"]] for m in pro_frontier], dtype=float)
    P = np.array([[x for x in pro_results[m]["pert"]] for m in pro_frontier], dtype=float)
    pro_gap = (O - P).mean(axis=0)

    # Per-item fidelity = orig vs pert correlation across models (item-level
    # agreement). For binary outcomes we use Pearson on per-model values.
    pro_fidelity = np.zeros(n_pro)
    for i in range(n_pro):
        oi = O[:, i]
        pi = P[:, i]
        if oi.std() == 0 or pi.std() == 0:
            pro_fidelity[i] = 1.0 if (oi == pi).all() else 0.0
        else:
            pro_fidelity[i] = np.corrcoef(oi, pi)[0, 1]

    valid = ~np.isnan(pro_fidelity)
    r_gap_fid = np.corrcoef(pro_gap[valid], pro_fidelity[valid])[0, 1]
    print(f"\nMMLU-PRO per-item correlation(gap, fidelity) = {r_gap_fid:+.3f}")
    print("  (artifact prediction: strongly negative; observed:",
          "near zero" if abs(r_gap_fid) < 0.3 else "non-trivial)")


if __name__ == "__main__":
    main()
