# Benchmark Half-Life — Code & Data Supplement

Reproduces all experiments and figures in *"The Half-Life of LLM Benchmarks: Discriminative Decay and Early Memorization Signals"* (NeurIPS 2026 Evaluations & Datasets Track submission).

## Quick Start: Compute Half-Life for Any Benchmark

```bash
pip install pandas scipy matplotlib
python -m halflife your_scores.csv --top-k 0.20 --n-boot 1000
# → "Discriminative half-life: 6.4 months [3.6, 9.7]"
```

Input CSV format: `model,score,eval_date` (one row per model evaluation).

## Repository structure

```
benchmark-halflife/
├── data/
│   ├── leaderboard/               # Aggregate leaderboard data
│   │   ├── *_v2.csv               # Open LLM Leaderboard v2 (3,977 unique models, 24,276 records)
│   │   └── glue_historical.csv    # GLUE retrospective (14 BERT-era models)
│   ├── perturbations/             # Item-level paraphrases
│   │   ├── mmlu_perturbed.json    # 228 items
│   │   ├── arc_perturbed.json     # 200 items
│   │   ├── gsm8k_perturbed.json   # 200 items
│   │   ├── truthfulqa_perturbed.json # 200 items (Claude-paraphrased)
│   │   ├── humaneval_perturbed.json  # 164 items (function rename + docstring rephrase)
│   │   └── mbpp_perturbed.json    # 200 items
│   ├── evaluations/               # ~13,000 perturbation-holdout evaluations
│   │   ├── mmlu_results.json      # 7 models × 228 × 2 versions
│   │   ├── arc_results.json
│   │   ├── gsm8k_results.json
│   │   ├── truthfulqa_results.json
│   │   ├── humaneval_results.json
│   │   └── mbpp_results.json
│   ├── decay/                     # Half-life fits per benchmark
│   ├── bootstrap/                 # 95% bootstrap CIs for half-life
│   ├── topk_sensitivity/          # Stratification sweep (k ∈ 5%–50%)
│   ├── model_selection/           # AIC/BIC for exp/stretched/logistic decay
│   ├── decomposition/             # G/K/F mechanism attribution
│   ├── akhtar_comparison/         # vs Akhtar et al. saturation index
│   ├── irt_full/                  # 1PL Rasch IRT fits
│   ├── item_drift/                # Item-level drift classification
│   ├── synthetic/                 # Single synthetic validation
│   ├── synthetic_sweep/           # 12 cells × 3 reps controlled sweep
│   ├── fidelity/                  # Perturbation fidelity correlation
│   ├── forecast/                  # Prospective forecast results
│   ├── triage/                    # Triage simulation + temporal contamination
│   ├── bootstrap_gaps/            # Bootstrap CIs on contamination gaps
│   ├── archaeology/               # BERT-era retrospective half-lives
│   └── meta_expanded/             # Meta-regression results (v1-only, cross-era, BERT-only)
├── src/                           # All scripts (see below)
├── halflife/                      # Reusable toolkit (python -m halflife)
│   ├── __init__.py
│   ├── core.py                    # compute_halflife(), fit_decay(), bootstrap_halflife()
│   └── __main__.py                # CLI entry point
├── figures/                       # Generated PDFs
├── release/                       # BCDD release files (per-benchmark JSON)
└── dataset_card.md                # HuggingFace dataset card (Croissant)
```

## Reproduction

### 0. Setup

```bash
pip install datasets pandas scipy matplotlib openai anthropic huggingface_hub
export OPENAI_API_KEY="<your-openai-api-key>"      # required for evaluation + paraphrase
export ANTHROPIC_API_KEY="<your-anthropic-api-key>"  # required for cross-vendor + paraphrase
```

### 1. Data collection

```bash
# Aggregate leaderboard data (no API key needed)
python src/collect_leaderboard.py

# v1 detail repos for full v1 history (uses HF unauthenticated API; rate-limited)
# script reads 7,055 archived per-model detail repos
# generates v1_scores_full.csv (~30 min)
```

### 2. Perturbation generation

```bash
# Multiple-choice and math benchmarks
python src/generate_perturbations.py all       # MMLU/ARC/GSM8K via OpenAI
python src/truthfulqa_claude_paraphrase.py     # TruthfulQA via Claude
python src/humaneval_pipeline.py perturb       # HumanEval via Claude
python src/mbpp_pipeline.py perturb            # MBPP via Claude
```

### 3. Model evaluation (cross-vendor)

```bash
# OpenAI evaluation
python src/evaluate_models.py all
python src/run_truthfulqa_full.py
python src/humaneval_pipeline.py eval
python src/humaneval_gpt4_only.py              # retry GPT-4 family with lower parallelism
python src/mbpp_pipeline.py eval
python src/add_gpt4o_mini.py                   # additional model

# Claude evaluation (cross-vendor replication)
python src/run_claude_models.py all            # MMLU/ARC/GSM8K
python src/run_truthfulqa_full.py              # already includes Claude
```

### 4. Analysis

```bash
# Half-life estimation with bootstrap CIs
python src/bootstrap_halflife.py

# Decay model selection (exp / stretched / logistic via AIC)
python src/model_selection.py

# Top-k stratification sensitivity (k ∈ 5%–50%)
python src/topk_sensitivity.py

# Mechanism attribution from perturbation holdouts
python src/decomposition_v2.py

# Item-level IRT drift (1PL Rasch)
python src/irt_full.py
python src/item_drift.py                       # response-pattern classifier

# Comparisons
python src/akhtar_comparison.py                # vs static saturation index
python src/fidelity_check.py                   # perturbation fidelity verification

# Bootstrap CIs on contamination gaps
python src/bootstrap_gaps.py

# Time-shifted contamination analysis
python src/time_shifted.py

# Synthetic validation
python src/synthetic_validation.py             # single setting
python src/synthetic_sweep.py                  # 36-run sweep

# Forecasting and decision tools
python src/prospective_forecast.py             # leave-last-quarter-out
python src/triage_simulation.py                # benchmark selection policy
python src/triage_bootstrap.py                 # bootstrap CIs for triage

# Benchmark archaeology (BERT-era retrospective half-lives)
python src/archaeology.py                      # GLUE/SuperGLUE/SQuAD/CoQA
python src/meta_regression_v2.py               # n=6 LLM + n=11 cross-era + n=5 BERT

# Temporal contamination fingerprint
python src/temporal_contamination.py           # per-item progressive contamination

# Prospective forecast validation on 2024Q2-2025Q1 v2 data
python src/forecast_validation.py

# MBPP retro-holdout redesign (identifier rename, Claude-only pilot)
python src/mbpp_retro_holdout.py
```

### 5. Paper PDF

The compiled paper PDF (main text + references + appendices, 35 pages) is available through the OpenReview submission and is not included in this anonymized code/data repository.

## Data sources

- **Open LLM Leaderboard v2**: HuggingFace dataset `open-llm-leaderboard/contents` (public)
- **Open LLM Leaderboard v1**: archived in `open-llm-leaderboard-old/details_*` repos (public, rate-limited)
- **MMLU items**: `cais/mmlu` (HuggingFace public)
- **GSM8K items**: `openai/gsm8k` (HuggingFace public)
- **ARC-Challenge items**: `allenai/ai2_arc` (HuggingFace public)
- **TruthfulQA items**: `truthful_qa` (HuggingFace public)
- **HumanEval items**: `openai_humaneval` (HuggingFace public)
- **MBPP items**: `mbpp/sanitized` (HuggingFace public)
- **GLUE historical**: hand-curated from BERT-era papers (`data/leaderboard/glue_historical.csv`)
- **Model evaluations**: OpenAI API (gpt-3.5-turbo-0125, gpt-4-turbo-2024-04-09, gpt-4o-2024-08-06, gpt-4o-mini-2024-07-18, gpt-4.1-2025-04-14) and Anthropic API (claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5).

## Compute budget

- **OpenAI API**: ≈ \$25–35 (evaluation calls + perturbation paraphrase)
- **Anthropic API**: ≈ \$15–25 (cross-vendor evaluation + Claude paraphrase)
- **CPU time**: ≈ 2 hours total for all analysis (no GPU required)
- **Storage**: ≈ 200 MB

## Key results (reproduced from paper)

### Half-life of LLM benchmarks

| Benchmark | Source | $\tau_{1/2}^{\text{exp}}$ [95% CI] | $\tau_{1/2}^{\text{best-AIC}}$ |
|-----------|--------|-------------|--------|
| WinoGrande | v1 | 1.0 [0.7, 2.2] mo | 1.7 (stretched exp) |
| HellaSwag | v1 | 2.5 [0.9, 6.7] mo | 2.5 (exponential) |
| ARC-Challenge | v1 | 2.7 [1.2, 4.9] mo | 3.4 (logistic) |
| GSM8K | v1 | 4.6 [2.8, 7.8] mo | 4.9 (logistic) |
| TruthfulQA | v1 | 9.0 [4.6, 19.2] mo | 5.7 (logistic) |
| BBH | v2 | 39.3 [34.7, 96.9] → **6.4 [3.6, 9.7]** mo (collapse) | 19.6 (logistic) |
| GLUE (retrospective) | hand | 4.9 mo | exponential |

### Cross-vendor mean contamination gap

| Benchmark | OpenAI (4 models) | Claude (3 models) | Ratio |
|-----------|:-:|:-:|:-:|
| MMLU | +0.016 | +0.028 | 1.7× |
| ARC-Challenge | +0.007 | +0.005 | 0.7× |
| **GSM8K** | **+0.134** | **+0.132** | **1.0× (industry-wide)** |
| TruthfulQA | +0.010 | +0.052 | 5.2× (Claude-specific) |
| HumanEval | +0.005 | +0.037 | 7.4× (Claude-specific) |

### Item-level FDR-corrected contamination signature

| Benchmark | Items with $\Delta b_i < -1$ (BH-FDR $q=0.05$) |
|-----------|:-:|
| GSM8K | **30 / 200 (15.0%)** |
| MMLU | 3 / 228 (1.3%) |
| ARC-C | 0 / 200 (0%) |

### Meta-regression (exploratory; treated as a negative result in the paper)

A log-linear fit on the v1 LLM-era subset (n=6) shows item-diversity (subtask count) correlates with half-life ($r=+0.85$, $p=0.030$, $R^2=0.78$). However, this relationship does **not** survive out-of-sample validation: adding v2 observed half-lives drops the correlation to $r=+0.41$ ($p=0.24$, n=10), and cross-era extension to BERT-era benchmarks gives near-zero correlation. We therefore retain this analysis only as exploratory and do not present diversity as a validated quantitative predictor of half-life. See Appendix M of the paper for full discussion.

### Triage simulation

| Policy | $k=1$ | $k=2$ | $k=3$ |
|--------|:-:|:-:|:-:|
| Random | 0.821 | 0.857 | 0.857 |
| Akhtar low | 0.893 | 0.893 | 0.893 |
| **Half-life long (ours)** | **0.893** | 0.821 | **0.964** |

Spearman ρ between policy ranking and oracle ranking on 7 frontier models.

## License

- **Code** (`src/`, `halflife/`): MIT.
- **BCDD perturbation pairs** (`release/benchmark_contamination_diagnostic_dataset/`): **CC BY 4.0**, the most permissive licence compatible with all source-benchmark licences. Original-item text is included verbatim under each source licence (GSM8K MIT, MMLU/MMLU-PRO MIT, ARC-Challenge CC BY-SA 4.0, TruthfulQA Apache 2.0, HumanEval MIT, MBPP CC BY 4.0); perturbed counterparts are derivative works owned by the authors and released under CC BY 4.0.
- Full Croissant metadata (core + Responsible AI fields) at `release/benchmark_contamination_diagnostic_dataset/croissant.json`.

## Citation

[anonymous for review]
