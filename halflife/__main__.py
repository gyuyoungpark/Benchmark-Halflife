"""CLI entry point: python -m halflife scores.csv [--top-k 0.20] [--n-boot 1000]"""
import argparse
import json
import sys
from halflife.core import compute_halflife


def main():
    parser = argparse.ArgumentParser(
        description="Compute discriminative half-life of an LLM benchmark from leaderboard scores."
    )
    parser.add_argument("csv", help="Path to CSV with columns: model, score, eval_date")
    parser.add_argument("--top-k", type=float, default=0.20, help="Top-k fraction (default: 0.20)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default: 1000)")
    parser.add_argument("--date-col", default="eval_date", help="Date column name")
    parser.add_argument("--score-col", default="score", help="Score column name")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    result = compute_halflife(
        args.csv,
        top_k=args.top_k,
        n_boot=args.n_boot,
        date_col=args.date_col,
        score_col=args.score_col,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if result["decay_detected"]:
            print(f"Discriminative half-life: {result['halflife_mo']:.1f} months")
            if result["ci_lo"] is not None:
                print(f"  95% bootstrap CI: [{result['ci_lo']:.1f}, {result['ci_hi']:.1f}] months")
        else:
            print("No decay detected (benchmark may still be in honeymoon phase)")
        print(f"  Quarters analyzed: {result['n_quarters']}")
        print(f"  Unique models: {result['n_models']}")


if __name__ == "__main__":
    main()
