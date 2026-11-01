import argparse
import json
import time
from pathlib import Path

import pandas as pd

from app.evaluation.experiments import (
    EVAL_SEED,
    aggregate_results,
    compare_agents,
    paired_differences,
)
from app.evaluation.visualize import plot_agent_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the symbolic baseline and the hybrid Wumpus agent.",
    )
    parser.add_argument("--worlds", type=int, default=2000, help="number of random worlds")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="evaluation seed")
    parser.add_argument(
        "--model",
        default="artifacts/pit_risk_logreg.joblib",
        help="path to a trained pit-risk model",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="directory for CSV, JSON, and PNG outputs",
    )
    parser.add_argument("--max-steps", type=int, default=100, help="max actions per episode")
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.6,
        help="hybrid pit-risk acceptance threshold",
    )
    parser.add_argument(
        "--utility-floor",
        type=float,
        default=-8.0,
        help="minimum hybrid utility for a risky frontier move",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found at {model_path}. "
            "Run `uv run python -m app.ml.train` first, or pass --model."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    df = compare_agents(
        pit_model_path=str(model_path),
        num_worlds=args.worlds,
        width=4,
        height=4,
        num_pits=3,
        seed=args.seed,
        max_steps=args.max_steps,
        risk_threshold=args.risk_threshold,
        utility_floor=args.utility_floor,
    )
    elapsed = time.perf_counter() - started

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    summary = aggregate_results(df)
    print(
        f"worlds={args.worlds} seed={args.seed} model={model_path} "
        f"max_steps={args.max_steps} risk_threshold={args.risk_threshold} "
        f"utility_floor={args.utility_floor} elapsed={elapsed:.1f}s"
    )
    print(df.head())
    print("\nSummary")
    print(summary.to_string(index=False))

    metrics = [
        "score",
        "steps",
        "survived",
        "escaped",
        "escaped_with_gold",
        "used_arrow",
        "risky_entries",
    ]
    paired = pd.DataFrame([paired_differences(df, metric) for metric in metrics])
    print("\nPaired hybrid - baseline (95% bootstrap CI)")
    print(paired.to_string(index=False))

    summary.to_csv(output_dir / "eval_summary.csv", index=False)
    paired.to_csv(output_dir / "paired_comparison.csv", index=False)
    config = {
        "worlds": args.worlds,
        "seed": args.seed,
        "model": str(model_path),
        "max_steps": args.max_steps,
        "risk_threshold": args.risk_threshold,
        "utility_floor": args.utility_floor,
        "elapsed_seconds": round(elapsed, 2),
    }
    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    plot_agent_comparison(df, output_dir=str(output_dir))


if __name__ == "__main__":
    main()
