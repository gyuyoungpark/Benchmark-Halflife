"""
halflife: A toolkit for measuring discriminative half-life of LLM benchmarks.

Usage:
    from halflife import compute_halflife
    result = compute_halflife("path/to/scores.csv")
    print(f"τ½ = {result['halflife_mo']:.1f} months [{result['ci_lo']:.1f}, {result['ci_hi']:.1f}]")

Or from CLI:
    python -m halflife path/to/scores.csv --top-k 0.20 --n-boot 1000
"""

__version__ = "0.1.0"

from halflife.core import compute_halflife, fit_decay, bootstrap_halflife
