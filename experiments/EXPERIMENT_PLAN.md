# Experiment Plan: Discriminative Half-Life of LLM Benchmarks

## Overview

3-phase pipeline: (1) Data Collection, (2) IRT + Decay Analysis, (3) Decomposition.
Target: 10 benchmarks, 3+ years of leaderboard data, item-level responses for IRT.

---

## Phase 1: Data Collection (Week 1)

### 1A. Aggregate Leaderboard Time Series

**Sources:**
- Open LLM Leaderboard v1 (HuggingFace): MMLU, ARC-C, HellaSwag, TruthfulQA, WinoGrande, GSM8K
- Open LLM Leaderboard v2: MMLU-Pro, GPQA, BBH, MATH, HumanEval
- Papers with Code SOTA tables
- Epoch AI benchmark tracker (epoch.ai/benchmarks)

**Collection method:**
- HuggingFace datasets API: `datasets.load_dataset("open-llm-leaderboard/results")`
- Papers with Code API: `GET /api/v1/papers/` with benchmark filter
- Epoch AI: CSV download from public data page
- For each (benchmark, model): record score, model name, model release date, evaluation date

**Target:** >=50 models per benchmark, spanning >=2 years

**Script:** `src/collect_leaderboard.py`

### 1B. Item-Level Response Matrices

**Critical for IRT analysis.** Sources:

| Benchmark | Item-level source | Feasibility |
|-----------|------------------|-------------|
| MMLU | lm-evaluation-harness logs on HF | High — many models have detailed logs |
| GSM8K | lm-eval-harness + custom eval | High — binary correct/incorrect |
| HumanEval | public pass@1 per-problem data | Medium — some models report per-problem |
| ARC-C | lm-eval-harness logs | High |
| HellaSwag | lm-eval-harness logs | High |
| TruthfulQA | lm-eval-harness logs | Medium |
| GPQA | Limited public data | Low — may need re-evaluation |
| WinoGrande | lm-eval-harness logs | High |

**Fallback plan:** For benchmarks without item-level data, re-evaluate representative models (~5 per epoch) using lm-evaluation-harness locally. Prioritize models that bracket the ability range in each epoch.

**Script:** `src/collect_itemlevel.py`

### 1C. Perturbation Holdouts (for Decomposition)

Generate for 3 benchmarks: MMLU, GSM8K, ARC-Challenge.

**MMLU perturbations:**
- Option shuffle (4! = 24 permutations, sample 1 random permutation per item)
- Stem paraphrasing via LLM (Claude/GPT-4) with human spot-check (10% sample)
- Distractor replacement (replace 1 wrong option with a new plausible distractor)

**GSM8K perturbations:**
- Number substitution (change numerical values, preserve solution structure)
- Name substitution
- Context rephrasing

**ARC-Challenge perturbations:**
- Option shuffle + stem paraphrasing

**Validation:** Perturbation holdouts must satisfy:
- Same difficulty distribution as originals (test via pilot with 5 models)
- <5% semantic drift (human evaluation on 100-item sample)

**Script:** `src/generate_perturbations.py`

---

## Phase 2: IRT + Decay Analysis (Week 2)

### 2A. Epoch Construction

- Partition timeline into **quarterly epochs** (Q1 2022, Q2 2022, ..., Q1 2026)
- Each epoch: models released in that quarter
- Minimum 5 models per epoch (merge adjacent quarters if needed)
- Record epoch membership in `data/epochs.json`

### 2B. IRT Fitting

**Tool:** `py-irt` (Python) or `mirt` (R via rpy2)

**Per epoch:**
1. Construct binary response matrix X_t (models x items)
2. Fit 2PL model: estimate (a_i, b_i) for each item, theta_j for each model
3. Check convergence (log-likelihood, RMSD)
4. Record all parameters

**Scale equating (Stocking-Lord):**
- Anchor models: select 3-5 models available across all epochs
  - Candidates: GPT-3.5-turbo, Llama-2-7B, Mistral-7B-v0.1
- Apply Stocking-Lord transformation to align (a, b) scales across epochs
- Implementation: `src/scale_equating.py`

**Output:** `data/irt_params/{benchmark}_epoch_{t}.json` with (a_i, b_i) per item per epoch

### 2C. Item Parameter Drift Detection

**Per item:**
1. Extract time series of (a_it, b_it) across epochs
2. Linear regression: a_it = alpha + beta * t + eps
   - Test H0: beta = 0 (no drift)
   - Bonferroni correction for multiple items
3. Lord's chi-squared test between first and last epoch
4. Classify items into drift patterns:
   - Stable: no significant drift in a or b
   - Difficulty-drifting: significant b decrease, a stable
   - Discrimination-collapsing: significant a decrease
   - Ceiling: both a and b decrease, >95% correct in recent epochs

**Script:** `src/drift_detection.py`

### 2D. Discriminative Power Metrics + Decay Fitting

**Compute per epoch per benchmark:**

```
sigma2_t = Var(scores_t)                              # inter-model variance
tau_t = KendallTau(rank_t, rank_{t+1})                # ranking stability
a_bar_t = mean(a_it for i in items)                   # aggregate IRT discrimination
C_t = 1 - IQR(top_k_scores_t) / IQR(all_scores_t)   # ceiling compression
```

**Decay curve fitting:**

For each (benchmark, metric) pair, fit 3 models via scipy.optimize.curve_fit:

1. Exponential: D_t = D0 * exp(-lambda * t) + D_inf
2. Stretched exponential: D_t = D0 * exp(-(lambda * t)^beta) + D_inf
3. Logistic decay: D_t = (D0 - D_inf) / (1 + exp(lambda * (t - t0))) + D_inf

Model selection: AIC, BIC. Report half-life from best model.

**Bootstrap CI:** 1000 iterations, resample models per epoch.

**Script:** `src/decay_fitting.py`

---

## Phase 3: Decomposition (Week 2-3)

### 3A. Evaluate Models on Perturbation Holdouts

- For each benchmark with perturbation holdouts (MMLU, GSM8K, ARC-C)
- Evaluate the same model set on both original and perturbation versions
- Use lm-evaluation-harness with custom task configs

**Script:** `src/eval_perturbations.py`

### 3B. Compute Decomposition

Per epoch per benchmark:
```
D_t_orig = discriminative_power(original_scores_t)
D_t_pert = discriminative_power(perturbation_scores_t)

K_t = D_t_pert - D_t_orig                    # contamination component
F_t = ceiling_information_loss(scores_t)      # design ceiling component
G_t = (D_0 - D_t) - K_t - F_t               # genuine convergence (residual)
```

### 3C. Synthetic Validation

Generate synthetic benchmark:
- 500 items, known (a_i, b_i) from U(0.5, 2.5) x N(0, 1)
- 20 epochs, theta ~ N(mu_t, sigma) with mu_t increasing linearly
- Inject contamination: fraction p_t of items known to fraction q_t of models
- Apply decomposition, compare estimated vs true G, K, F
- Sweep over contamination rates to test sensitivity

**Script:** `src/synthetic_validation.py`

---

## Phase 4: Analysis and Writing (Week 3)

### Key Figures

1. **Half-life table:** Benchmark x metric matrix with half-life estimates and 95% CI
2. **Decay curves:** Per benchmark, 4 metrics over time with fitted curves (2x5 subplot grid)
3. **IRT drift heatmap:** Items x epochs, color = a_i change (for MMLU as case study)
4. **Drift pattern pie charts:** Fraction of items in each category per benchmark
5. **Decomposition stacked area:** G_t + K_t + F_t over time for MMLU, GSM8K, ARC-C
6. **Synthetic validation:** Estimated vs true components (scatter + R^2)
7. **Predictor regression:** Half-life vs benchmark properties (scatter + regression line)

### Key Tables

1. Benchmark statistics (items, age, models, etc.)
2. Half-life estimates (all benchmarks x all metrics)
3. Drift detection summary (% items in each drift category)
4. Decomposition results at latest epoch
5. Model selection (AIC/BIC for 3 decay models)

---

## Computational Requirements

| Task | Compute | Time estimate |
|------|---------|---------------|
| Leaderboard scraping | CPU only | ~2 hours |
| Item-level collection | CPU + network | ~1 day |
| IRT fitting (per benchmark per epoch) | CPU, ~10 min | ~10 benchmarks x 16 epochs = ~27 hours |
| Perturbation generation | LLM API calls | ~$50-100 API cost |
| Model re-evaluation (if needed) | GPU (A100) | ~2-3 days for 50 models x 3 benchmarks |
| Decay fitting + bootstrap | CPU | ~2 hours |
| Synthetic validation | CPU | ~1 hour |

**Total estimated cost:** ~$100-200 API + 3-5 days compute

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Insufficient item-level data | Fallback: re-evaluate 5 models/epoch via lm-eval-harness |
| Too few models per epoch | Merge adjacent quarters; use 6-month windows |
| IRT non-convergence | Reduce to Rasch (1PL) for problematic benchmarks |
| Perturbation difficulty mismatch | Pilot test + calibrate before full evaluation |
| Scale equating instability | Test multiple anchor model sets; report sensitivity |
