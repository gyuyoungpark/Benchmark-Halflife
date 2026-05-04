# GSM8K 125-item verified subset — manual spot-check protocol (30 items, 2 annotators)

**Purpose.** §4 of the paper claims that on the 125-item programmatically-verified equivalence subset, the orig−pert gap is $+8.6$pp [+4.3, +13.3] (CI excluding zero). To strengthen this from "programmatically verified" to "programmatically + human-verified", manually inspect 30 items from the verified set with two independent annotators.

**Subset items file:** `verified_subset_items.json` (50 items, fixed seed 42, drawn from the 125 verified pool). Use `verified_subset.json["verified_idx"]` to access the full 125-item pool if you want to draw a fresh 30 from the full pool.

**Recommended protocol (≈90 minutes per annotator).**
1. Two annotators independently inspect 30 items from the verified pool. Use seed 42 for the first 30 of `verified_subset.json["headline_subset_idx"]` to keep deterministic.
2. For each item, score on three dimensions:
   - **Equivalent operation type?** (yes/no) — same add/subtract/multiply/divide chain
   - **Same step count?** (yes/no) — same number of intermediate computations  
   - **Comparable arithmetic complexity?** (yes/no) — digit length, carry/borrow count, divisibility structure broadly similar
3. Mark each item as **equivalent** (3/3 yes), **partial** (2/3 yes), or **non-equivalent** (≤1/3 yes).
4. Compute Cohen's κ between the two annotators on the 3-class label.

**Target for paper update:** ${\ge}27/30$ items rated **equivalent** by both annotators, with κ ${\ge}0.6$.

**If hit:** §4 fidelity caveat can be strengthened to "two independent annotators rated 27 of 30 randomly drawn items as equivalent (Cohen's κ = 0.X), supporting the programmatic verification".

**If miss (e.g., only 20/30 equivalent):** report the actual rate honestly. The +8.6pp claim still holds since CI excludes zero on the full 125 set; the paper text would simply qualify as "programmatic verification + 30-item human spot-check yielded 20/30 fully-equivalent items, with the remaining 10 marked partial; treating those 10 as non-equivalent and rerunning the gap on the strict 115-item subset still yields +X.Xpp [CI ...]".

**Output:** Save annotations as `human_spotcheck_30.json`:
```json
{
  "annotators": ["annotator_a", "annotator_b"],
  "date": "2026-05-04",
  "items_checked": [<list of 30 item indices>],
  "annotator_1": [{"idx": 47, "ops": true, "steps": true, "complexity": true, "label": "equivalent"}, ...],
  "annotator_2": [...],
  "agreement_count_equivalent": 27,
  "cohen_kappa": 0.72,
  "notes": "..."
}
```

**Then update §4 to add:** "Two independent annotators (Cohen's κ = 0.72) rated 27 of 30 items from the verified pool as fully equivalent on operation type, step count, and arithmetic complexity. On the 27-item strict-equivalent subset, the gap is $+X$pp [CI ...] (or: rerunning on the full 125 set with the 3 partial-equivalents excluded gives $+Y$pp [CI ...])."

**Time budget for this:** ~90 min/annotator × 2 + 30 min adjudication = 3.5 hours total. Doable in one afternoon before May 6 deadline.
