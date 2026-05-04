"""
Collect aggregate leaderboard time series from Open LLM Leaderboard (HuggingFace).
Outputs: data/leaderboard/{benchmark}_scores.csv with columns: model, score, release_date, eval_date
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# Target benchmarks and their HF dataset identifiers
BENCHMARKS_V1 = {
    "mmlu": "hendrycksTest",
    "arc_challenge": "arc:challenge",
    "hellaswag": "hellaswag",
    "truthfulqa": "truthfulqa:mc2",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
}

BENCHMARKS_V2 = {
    "mmlu_pro": "mmlu_pro",
    "gpqa": "gpqa",
    "bbh": "bbh",
    "math": "math",
    "humaneval": "humaneval",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "leaderboard"


def collect_from_hf_api(output_dir: Path):
    """Collect from HuggingFace Open LLM Leaderboard datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open LLM Leaderboard v1 results
    print("Loading Open LLM Leaderboard v1...")
    try:
        ds = load_dataset("open-llm-leaderboard/results", split="train")
        df = ds.to_pandas()
        print(f"  Loaded {len(df)} entries from v1")

        # Extract per-benchmark scores
        for bench_name, bench_key in BENCHMARKS_V1.items():
            if bench_key in df.columns:
                bench_df = df[["model_name", bench_key, "submission_date"]].copy()
                bench_df.columns = ["model", "score", "eval_date"]
                bench_df = bench_df.dropna(subset=["score"])
                bench_df.to_csv(output_dir / f"{bench_name}_scores.csv", index=False)
                print(f"  {bench_name}: {len(bench_df)} entries")
            else:
                print(f"  {bench_name}: column '{bench_key}' not found, available: {list(df.columns)[:10]}...")
    except Exception as e:
        print(f"  v1 failed: {e}")
        print("  Falling back to manual collection strategy...")
        collect_fallback(output_dir)


def collect_fallback(output_dir: Path):
    """Fallback: collect from Papers with Code API or cached data."""
    print("\nFallback collection strategy:")
    print("  1. Download from Papers with Code: https://paperswithcode.com/api/v1/")
    print("  2. Download from Epoch AI: https://epoch.ai/data/benchmarks")
    print("  3. Scrape HF model cards for individual benchmark scores")
    print("\nManual steps:")

    # Generate template CSVs for manual filling
    for bench_name in list(BENCHMARKS_V1.keys()) + list(BENCHMARKS_V2.keys()):
        template = pd.DataFrame(columns=["model", "score", "release_date", "eval_date"])
        path = output_dir / f"{bench_name}_scores.csv"
        if not path.exists():
            template.to_csv(path, index=False)
            print(f"  Created template: {path}")


def collect_from_pwc_api(output_dir: Path):
    """Collect SOTA results from Papers with Code API."""
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)

    PWC_BENCHMARKS = {
        "mmlu": "massive-multitask-language-understanding",
        "gsm8k": "grade-school-math-8k",
        "humaneval": "code-generation-on-humaneval",
        "arc_challenge": "ai2-arc-challenge",
        "hellaswag": "hellaswag",
    }

    for bench_name, pwc_slug in PWC_BENCHMARKS.items():
        print(f"Collecting {bench_name} from Papers with Code...")
        url = f"https://paperswithcode.com/api/v1/tasks/{pwc_slug}/results/"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                results = resp.json()
                rows = []
                for r in results.get("results", []):
                    rows.append({
                        "model": r.get("model", ""),
                        "score": r.get("value", None),
                        "release_date": r.get("paper_date", ""),
                        "eval_date": r.get("evaluated_date", ""),
                    })
                df = pd.DataFrame(rows)
                df.to_csv(output_dir / f"{bench_name}_pwc.csv", index=False)
                print(f"  {bench_name}: {len(df)} entries")
            else:
                print(f"  {bench_name}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  {bench_name}: {e}")


def merge_sources(output_dir: Path):
    """Merge data from multiple sources into unified per-benchmark CSVs."""
    for bench_name in list(BENCHMARKS_V1.keys()) + list(BENCHMARKS_V2.keys()):
        dfs = []
        for suffix in ["_scores.csv", "_pwc.csv", "_epoch.csv"]:
            path = output_dir / f"{bench_name}{suffix}"
            if path.exists():
                df = pd.read_csv(path)
                df["source"] = suffix.replace("_", "").replace(".csv", "")
                dfs.append(df)

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            # Deduplicate by model name, keeping highest-source-priority
            merged = merged.drop_duplicates(subset=["model"], keep="first")
            merged = merged.sort_values("release_date")
            merged.to_csv(output_dir / f"{bench_name}_merged.csv", index=False)
            print(f"{bench_name}: {len(merged)} unique models after merge")


def main():
    parser = argparse.ArgumentParser(description="Collect LLM benchmark leaderboard data")
    parser.add_argument("--source", choices=["hf", "pwc", "merge", "all"], default="all")
    parser.add_argument("--output", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.source in ("hf", "all"):
        collect_from_hf_api(output_dir)
    if args.source in ("pwc", "all"):
        collect_from_pwc_api(output_dir)
    if args.source in ("merge", "all"):
        merge_sources(output_dir)


if __name__ == "__main__":
    main()
