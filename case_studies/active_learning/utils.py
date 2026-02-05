import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_config import DATASETS

log = logging.getLogger(__name__)

# Default output directory (can be overridden via --output-dir)
RESULTS_DIR = Path.cwd() / "results"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active Learning Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 5 repeats with different seed draws
  uv run python -m experiment_runner --repeats 5

  # Quick test (1 repeat, 3 iterations)
  uv run python -m experiment_runner --repeats 1 --iterations 3

  # Custom parameters
  uv run python -m experiment_runner \\
      --repeats 10 --seed-size 500 --query-size 50 --iterations 20

  # From JSON config
  uv run python -m experiment_runner --config config.json
""",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="Path to JSON config file. If provided, other args are ignored.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="dbpedia_tiny",
        choices=list(DATASETS.keys()),
        help="Dataset to use (default: dbpedia_tiny)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of different seed draws to run (default: 5)",
    )
    parser.add_argument(
        "--seed-size",
        type=int,
        default=700,
        help="Initial labeled examples (default: 700)",
    )
    parser.add_argument(
        "--query-size",
        type=int,
        default=20,
        help="Examples to query per iteration (default: 20)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of AL iterations (default: 10)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Base random state for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="",
        help="Version tag for filtering results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from existing experiment file (adds more repeats)",
    )
    return parser.parse_args()


def _get_balanced_seed_indices(
    df: pd.DataFrame,
    seed_size: int,
    random_state: int,
) -> list[int]:
    """Sample balanced seed indices across classes.

    Args:
        df: DataFrame with 'label' column
        seed_size: Total number of seed samples to select
        random_state: Random state for reproducibility

    Returns:
        List of indices for seed samples
    """
    labels = df["label"].unique()
    n_classes = len(labels)
    per_class = seed_size // n_classes
    remainder = seed_size % n_classes

    indices = []
    rng = np.random.default_rng(random_state)

    for i, label in enumerate(sorted(labels)):
        class_indices = df[df["label"] == label].index.tolist()
        n_samples = per_class + (1 if i < remainder else 0)
        sampled = rng.choice(
            class_indices, size=min(n_samples, len(class_indices)), replace=False
        )
        indices.extend(sampled.tolist())

    if len(indices) < seed_size:
        log.warning(
            "Only %d seed samples available (requested %d)", len(indices), seed_size
        )

    return indices
