import uuid
from enum import Enum

from pydantic import BaseModel, Field


class OracleType(str, Enum):
    LLM = "llm"
    GROUND_TRUTH = "ground_truth"


class OracleLabelResult(BaseModel):
    label: int | None = Field(description="Label ID (None if failed)")
    category: str = Field(description="Category name")


class OracleQueryResult(BaseModel):
    labels: list[OracleLabelResult]
    session_url: str
    session_id: str
    latency_seconds: float | None = None
    cost_usd: float | None = None


class IterationResult(BaseModel):
    iteration: int  # 0 = after seed, 1 = after first query, etc.
    n_labeled: int
    accuracy: float
    f1_macro: float
    oracle_queried: int | None = None  # Total samples queried from oracle
    oracle_valid: int | None = None  # Samples with valid labels (label >= 0)
    oracle_failures: int | None = None  # Samples where oracle failed (label == -1)
    oracle_accuracy: float | None = None  # Accuracy on valid samples
    session_url: str | None = None
    session_id: str | None = None
    query_latency_seconds: float | None = None  # Time taken for LLM query
    query_cost_usd: float | None = None  # Cost of LLM query


class ExperimentConfig(BaseModel, frozen=True):
    """Configuration for repeated random split experiments.

    Instead of K-fold CV, we use a fixed train/test split and vary
    only the seed dataset draw across repeats. This gives cleaner
    comparison between GT and LLM oracles.
    """

    dataset: str = Field(description="Dataset name")
    test_fraction: float = Field(
        default=0.2, description="Fraction of data to hold out for testing"
    )
    repeats: int = Field(default=5, description="Number of different seed draws to run")
    seed_size: int = Field(description="Initial labeled examples")
    query_size: int = Field(description="Examples to query per iteration")
    iterations: int = Field(
        description="Number of AL query rounds (total evaluations = iterations + 1)"
    )
    random_state: int = Field(default=42)
    version: str = Field(default="")


class RepeatResult(BaseModel):
    """Result for a single repeat (one seed draw)."""

    repeat: int = Field(description="Repeat index (0-based)")
    seed_random_state: int = Field(description="Random state used for this seed draw")
    train_size: int
    test_size: int
    ground_truth_history: list[IterationResult]
    llm_history: list[IterationResult]
    ground_truth_final_accuracy: float
    ground_truth_final_f1: float
    llm_final_accuracy: float
    llm_final_f1: float
    llm_oracle_accuracy: float  # How accurate LLM was vs ground truth
    session_ids: list[str]  # For cost tracking


class ExperimentResult(BaseModel):
    """Result for a full experiment (multiple repeats)."""

    config: ExperimentConfig
    repeat_results: list[RepeatResult]
    repeats_completed: int
    mean_ground_truth_accuracy: float
    std_ground_truth_accuracy: float
    mean_llm_accuracy: float
    std_llm_accuracy: float
    mean_llm_oracle_accuracy: float
    total_duration_seconds: float
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
