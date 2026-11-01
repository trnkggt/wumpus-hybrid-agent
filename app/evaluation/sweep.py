"""Validation sweep for the hybrid risk threshold and utility floor.

This deliberately uses a seed distinct from both the training seed (42) and
the reported evaluation seed (2025).  Tune here, then report on the untouched
evaluation seed.

Usage:
    uv run python -m app.evaluation.sweep --worlds 500 --seed 777
"""

import argparse
from pathlib import Path

import pandas as pd

from app.evaluation.experiments import aggregate_results, compare_agents
from app.evaluation.visualize import plot_risk_return

VALIDATION_SEED = 777


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep hybrid risk settings.")
    parser.add_argument("--worlds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=VALIDATION_SEED)
    parser.add_argument("--thresholds", default="0.6,0.7,0.8,0.9")
    parser.add_argument("--utility-floor", type=float, default=-8.0)
    parser.add_argument("--model", default="artifacts/pit_risk_logreg.joblib")
    parser.add_argument("--output", default="artifacts/threshold_sweep.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]

    rows: list[dict] = []
    for threshold in thresholds:
        df = compare_agents(
            pit_model_path=args.model,
            num_worlds=args.worlds,
            seed=args.seed,
            risk_threshold=threshold,
            utility_floor=args.utility_floor,
        )
        summary = aggregate_results(df).set_index("agent")
        rows.append({
            "risk_threshold": threshold,
            "utility_floor": args.utility_floor,
            "baseline_score": summary.loc["baseline", "average_score"],
            "hybrid_score": summary.loc["hybrid", "average_score"],
            "baseline_gold_rate": summary.loc["baseline", "gold_rate"],
            "hybrid_gold_rate": summary.loc["hybrid", "gold_rate"],
            "hybrid_death_rate": summary.loc["hybrid", "death_rate"],
            "hybrid_risky_entries": summary.loc["hybrid", "average_risky_entries"],
        })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    plot_risk_return(result, output_dir=str(output.parent))


if __name__ == "__main__":
    main()
