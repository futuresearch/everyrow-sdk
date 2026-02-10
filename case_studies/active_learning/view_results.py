#!/usr/bin/env python
"""View and plot active learning experiment results.

This script loads pre-computed experiment results and generates plots
comparing ground truth vs LLM oracle performance.

Usage (from case_studies/active_learning/):
    # View results and generate plots
    uv run python -m view_results --results-dir ./results

    # Filter by version
    uv run python -m view_results --results-dir ./results --version my_experiment

    # Show plots interactively instead of just saving
    uv run python -m view_results --results-dir ./results --show
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_result(result_path: Path) -> dict:
    """Load experiment result from JSON file."""
    return json.loads(result_path.read_text())


def load_all_results(
    results_dir: Path,
    version: str | None = None,
) -> list[dict]:
    """Load all experiments from a results directory."""
    results = []

    for f in sorted(results_dir.glob("experiment_*.json")):
        try:
            data = load_result(f)
            if version and data["config"].get("version") != version:
                continue
            data["_filename"] = f.name
            results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {f.name}: {e}")
            continue

    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert experiment results to a pandas DataFrame for plotting."""
    rows = []

    for result in results:
        config = result["config"]
        experiment_id = result.get("experiment_id", "unknown")

        for repeat in result.get("repeat_results", []):
            # Ground truth history
            for h in repeat["ground_truth_history"]:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": config["dataset"],
                        "oracle_type": "ground_truth",
                        "version": config.get("version", ""),
                        "repeat": repeat["repeat"],
                        "iteration": h["iteration"],
                        "n_labeled": h["n_labeled"],
                        "accuracy": h["accuracy"],
                        "f1_macro": h["f1_macro"],
                    }
                )

            # LLM history
            for h in repeat["llm_history"]:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "dataset": config["dataset"],
                        "oracle_type": "llm",
                        "version": config.get("version", ""),
                        "repeat": repeat["repeat"],
                        "iteration": h["iteration"],
                        "n_labeled": h["n_labeled"],
                        "accuracy": h["accuracy"],
                        "f1_macro": h["f1_macro"],
                        "oracle_accuracy": h.get("oracle_accuracy"),
                    }
                )

    return pd.DataFrame(rows)


def plot_learning_curves(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Plot learning curves comparing ground truth vs LLM oracle."""

    output_dir.mkdir(parents=True, exist_ok=True)

    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by oracle type and n_labeled
    grouped = (
        df.groupby(["oracle_type", "n_labeled"])
        .agg(
            {
                "accuracy": ["mean", "std"],
                "f1_macro": ["mean", "std"],
            }
        )
        .reset_index()
    )
    grouped.columns = [
        "oracle_type",
        "n_labeled",
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
    ]

    colors = {"ground_truth": "#2ecc71", "llm": "#3498db"}
    labels = {"ground_truth": "Ground Truth Oracle", "llm": "LLM Oracle"}

    # Plot accuracy
    ax = axes[0]
    for oracle_type, group in grouped.groupby("oracle_type"):
        color = colors.get(oracle_type, "#333")
        label = labels.get(oracle_type, oracle_type)
        ax.plot(
            group["n_labeled"],
            group["accuracy_mean"],
            marker="o",
            label=label,
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            group["n_labeled"],
            group["accuracy_mean"] - group["accuracy_std"],
            group["accuracy_mean"] + group["accuracy_std"],
            alpha=0.2,
            color=color,
        )
    ax.set_xlabel("Number of Labeled Examples")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Learning Curve: Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot F1
    ax = axes[1]
    for oracle_type, group in grouped.groupby("oracle_type"):
        color = colors.get(oracle_type, "#333")
        label = labels.get(oracle_type, oracle_type)
        ax.plot(
            group["n_labeled"],
            group["f1_mean"],
            marker="o",
            label=label,
            color=color,
            linewidth=2,
        )
        ax.fill_between(
            group["n_labeled"],
            group["f1_mean"] - group["f1_std"],
            group["f1_mean"] + group["f1_std"],
            alpha=0.2,
            color=color,
        )
    ax.set_xlabel("Number of Labeled Examples")
    ax.set_ylabel("Test F1 Macro")
    ax.set_title("Learning Curve: F1 Macro")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.savefig(output_dir / "learning_curves.pdf")
    print(f"Saved: {output_dir / 'learning_curves.png'}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_learning_curve_accuracy(
    df: pd.DataFrame,
    results: list[dict],
    output_dir: Path,
    show: bool = False,
) -> None:
    """Plot accuracy learning curve with summary caption."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by oracle type and n_labeled
    grouped = (
        df.groupby(["oracle_type", "n_labeled"])
        .agg({"accuracy": ["mean", "std"]})
        .reset_index()
    )
    grouped.columns = ["oracle_type", "n_labeled", "accuracy_mean", "accuracy_std"]

    colors = {"ground_truth": "#2ecc71", "llm": "#3498db"}
    labels = {"ground_truth": "Ground Truth Oracle", "llm": "LLM Oracle"}
    markers = {"ground_truth": "o", "llm": "s"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for oracle_type, group in grouped.groupby("oracle_type"):
        color = colors.get(oracle_type, "#333")
        label = labels.get(oracle_type, oracle_type)
        marker = markers.get(oracle_type, "o")
        ax.plot(
            group["n_labeled"],
            group["accuracy_mean"],
            marker=marker,
            label=label,
            color=color,
            linewidth=2,
            markersize=6,
        )
        ax.fill_between(
            group["n_labeled"],
            group["accuracy_mean"] - group["accuracy_std"],
            group["accuracy_mean"] + group["accuracy_std"],
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Number of Labeled Examples", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Active Learning: Ground Truth vs LLM Oracle", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Calculate stats for caption
    total_repeats = sum(len(r.get("repeat_results", [])) for r in results)

    # Samples annotated per run
    if results and results[0].get("repeat_results"):
        first_repeat = results[0]["repeat_results"][0]
        llm_history = first_repeat.get("llm_history", [])
        samples_per_run = sum(h.get("oracle_queried", 0) or 0 for h in llm_history)
    else:
        samples_per_run = 0

    # LLM oracle accuracy
    llm_oracle_accs = []
    for r in results:
        for repeat in r.get("repeat_results", []):
            if "llm_oracle_accuracy" in repeat:
                llm_oracle_accs.append(repeat["llm_oracle_accuracy"])
    llm_oracle_mean = np.mean(llm_oracle_accs) if llm_oracle_accs else 0
    llm_oracle_std = np.std(llm_oracle_accs) if llm_oracle_accs else 0

    # Build caption
    caption_parts = []
    caption_parts.append(f"Samples annotated: {samples_per_run}")
    caption_parts.append(f"Runs: {total_repeats}")
    if llm_oracle_accs:
        caption_parts.append(
            f"LLM accuracy: {llm_oracle_mean * 100:.1f}% ± {llm_oracle_std * 100:.1f}%"
        )
    caption = "  |  ".join(caption_parts)

    fig.text(
        0.5,
        0.02,
        caption,
        ha="center",
        fontsize=10,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(
        output_dir / "learning_curve_accuracy.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(output_dir / "learning_curve_accuracy.pdf", bbox_inches="tight")
    print(f"Saved: {output_dir / 'learning_curve_accuracy.png'}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_oracle_accuracy(
    df: pd.DataFrame,
    output_dir: Path,
    show: bool = False,
) -> None:
    """Plot LLM oracle accuracy over iterations."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter to LLM oracle only
    llm_df = df[df["oracle_type"] == "llm"].copy()
    if llm_df.empty:
        print("No LLM oracle data for oracle accuracy plot")
        return

    # Group by iteration
    grouped = (
        llm_df.groupby("iteration")
        .agg({"oracle_accuracy": ["mean", "std", "count"]})
        .reset_index()
    )
    grouped.columns = ["iteration", "mean", "std", "count"]
    grouped = grouped[grouped["count"] > 0]

    if grouped.empty or grouped["mean"].isna().all():
        print("No oracle accuracy data available")
        return

    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        grouped["iteration"],
        grouped["mean"],
        marker="o",
        color="#3498db",
        linewidth=2,
        label="LLM Oracle Accuracy",
    )
    ax.fill_between(
        grouped["iteration"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        alpha=0.2,
        color="#3498db",
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Oracle Accuracy (vs Ground Truth)")
    ax.set_title("LLM Oracle Accuracy Over Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "oracle_accuracy.png", dpi=150)
    plt.savefig(output_dir / "oracle_accuracy.pdf")
    print(f"Saved: {output_dir / 'oracle_accuracy.png'}")

    if show:
        plt.show()
    else:
        plt.close()


def print_summary(results: list[dict]) -> None:
    """Print summary statistics for experiments."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for result in results:
        config = result["config"]
        repeats = len(result.get("repeat_results", []))

        print(f"\nExperiment: {result.get('experiment_id', 'unknown')}")
        print(f"  Dataset: {config['dataset']}")
        print(f"  Repeats: {repeats}")

        if "mean_ground_truth_accuracy" in result:
            print(
                f"  Ground Truth Accuracy: "
                f"{result['mean_ground_truth_accuracy']:.3f} ± "
                f"{result['std_ground_truth_accuracy']:.3f}"
            )
        if "mean_llm_accuracy" in result:
            print(
                f"  LLM Oracle Accuracy:   "
                f"{result['mean_llm_accuracy']:.3f} ± "
                f"{result['std_llm_accuracy']:.3f}"
            )
        if "mean_llm_oracle_accuracy" in result:
            print(f"  LLM Label Accuracy:    {result['mean_llm_oracle_accuracy']:.1%}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="View and plot active learning experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from case_studies/active_learning/):
  # Basic usage - load results and generate plots
  uv run python -m view_results

  # Specify results directory
  uv run python -m view_results --results-dir ./my_results

  # Show plots interactively
  uv run python -m view_results --show

  # Filter by experiment version
  uv run python -m view_results --version my_experiment
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path.cwd() / "results",
        help="Directory containing result JSON files (default: ./results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "plots",
        help="Directory to save plots (default: ./plots)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Filter experiments by version tag",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of just saving",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation, only print summary",
    )
    args = parser.parse_args()

    print(f"Loading experiments from: {args.results_dir}")
    results = load_all_results(args.results_dir, version=args.version)

    if not results:
        print("No experiments found!")
        print(f"Make sure {args.results_dir} contains experiment_*.json files")
        return 1

    print(f"Loaded {len(results)} experiment(s)")

    # Print summary
    print_summary(results)

    if args.no_plots:
        return 0

    # Convert to DataFrame for plotting
    df = results_to_dataframe(results)
    print(f"\nDataFrame shape: {df.shape}")

    # Save DataFrame
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_learning_curves(df, args.output_dir, show=args.show)
    plot_learning_curve_accuracy(df, results, args.output_dir, show=args.show)
    plot_oracle_accuracy(df, args.output_dir, show=args.show)

    print(f"\nPlots saved to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
