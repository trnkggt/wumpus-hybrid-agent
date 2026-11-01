from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Light chart palette from the project's data-viz reference.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
    ax.tick_params(colors=SECONDARY, labelsize=9)


def plot_agent_comparison(results_df: pd.DataFrame, output_dir: str = "artifacts") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = results_df.groupby("agent").agg(
        average_score=("score", "mean"),
        gold_rate=("escaped_with_gold", "mean"),
        survival_rate=("survived", "mean"),
    )

    for metric in ["average_score", "gold_rate", "survival_rate"]:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=SURFACE)
        summary[metric].plot(kind="bar", ax=ax, color=SERIES, legend=False)
        fmt = "{:.1%}" if metric in {"gold_rate", "survival_rate"} else "{:.1f}"
        for index, value in enumerate(summary[metric]):
            ax.annotate(
                fmt.format(value),
                (index, value),
                ha="center",
                va="bottom",
                fontsize=9,
                color=SECONDARY,
            )
        ax.set_title(metric.replace("_", " ").title(), color=INK)
        ax.set_ylabel(metric, color=SECONDARY)
        ax.set_xlabel("")
        _style_axis(ax)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=150, facecolor=SURFACE)
        plt.close(fig)


def plot_risk_return(sweep_df: pd.DataFrame, output_dir: str = "artifacts") -> None:
    """Plot the hybrid's gold/score trade-off against its death rate.

    ``sweep_df`` is the CSV produced by ``app.evaluation.sweep``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = sweep_df.sort_values("risk_threshold").reset_index(drop=True)
    x = df["hybrid_death_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
    panels = [
        (axes[0], "hybrid_gold_rate", "baseline_gold_rate", "Gold rate", "{:.0%}", "{:.1%}"),
        (axes[1], "hybrid_score", "baseline_score", "Mean score", "{:.0f}", "{:.0f}"),
    ]

    for ax, hybrid_col, baseline_col, ylabel, tick_fmt, value_fmt in panels:
        baseline = float(df[baseline_col].iloc[0])
        hybrid = df[hybrid_col]

        ax.plot(
            x,
            hybrid,
            color=SERIES,
            linewidth=2,
            marker="o",
            markersize=8,
            zorder=3,
            label="Hybrid",
        )
        ax.scatter(
            [0],
            [baseline],
            color=MUTED,
            marker="D",
            s=60,
            zorder=4,
            label="Baseline",
        )
        ax.annotate(
            "baseline",
            (0, baseline),
            xytext=(8, -14),
            textcoords="offset points",
            color=SECONDARY,
            fontsize=9,
        )

        # Label the high-risk end; summarise the tight safe cluster in one
        # callout so the low-risk points do not collide.
        cliff = df[df["risk_threshold"] == 0.80]
        if not cliff.empty:
            row = cliff.iloc[0]
            ax.annotate(
                "0.80+",
                (row["hybrid_death_rate"], float(row[hybrid_col])),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color=SERIES,
                fontsize=9,
            )

        safe = df[df["risk_threshold"] < 0.80]
        if safe.empty:
            safe = df
        if hybrid_col == "hybrid_gold_rate":
            callout = (
                f"threshold < 0.80\n"
                f"{safe[hybrid_col].min():.1%}-{safe[hybrid_col].max():.1%} gold\n"
                f"<= {safe['hybrid_death_rate'].max():.1%} deaths"
            )
            callout_y = 0.326
        else:
            callout = (
                f"threshold < 0.80\n"
                f"score {safe[hybrid_col].min():.0f}-{safe[hybrid_col].max():.0f}"
            )
            callout_y = 268
        ax.text(
            0.030,
            callout_y,
            callout,
            ha="left",
            va="center",
            fontsize=8.5,
            color=SECONDARY,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": SURFACE,
                "edgecolor": GRID,
                "linewidth": 0.8,
            },
        )

        ax.set_xlabel("Death rate", color=SECONDARY)
        ax.set_ylabel(ylabel, color=SECONDARY)
        ax.set_xlim(-0.008, max(float(x.max()), 0.22) * 1.1)
        if hybrid_col == "hybrid_gold_rate":
            low = min(float(hybrid.min()), baseline) - 0.02
            high = max(float(hybrid.max()), baseline) + 0.02
        else:
            low = 0
            high = max(float(hybrid.max()), baseline) * 1.12
        ax.set_ylim(low, high)
        _style_axis(ax)
        ax.legend(frameon=False, loc="best", fontsize=9, labelcolor=SECONDARY)

    fig.suptitle(
        "Hybrid risk-return trade-off (validation: 500 worlds, seed 777)",
        color=INK,
        fontsize=12,
    )
    threshold_low = float(df["risk_threshold"].min())
    threshold_high = float(df["risk_threshold"].max())
    utility_floor = float(df["utility_floor"].iloc[0])
    fig.text(
        0.5,
        0.01,
        f"Risk threshold {threshold_low:.2f}-{threshold_high:.2f}, utility floor "
        f"{utility_floor:.1f}. Default 0.60 (safe cluster); 0.80+ is the high-risk end.",
        ha="center",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out / "risk_return.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
