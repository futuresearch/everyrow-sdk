"""Active Learning Experiment Runner using EveryRow SDK.

Compares LLM oracle vs ground truth oracle using repeated random seed draws.
Uses a fixed train/test split and varies only the seed dataset across repeats,
giving a cleaner comparison between oracles.

Usage (from case_studies/active_learning/):
    # Run 5 repeats (different seed draws)
    EVERYROW_API_KEY=<your-api-key> uv run \
        python -m experiment_runner \
        --repeats 5 --seed-size 700 --query-size 20 --iterations 10

    # Quick test (1 repeat)
    EVERYROW_API_KEY=<your-api-key> uv run \
        python -m experiment_runner --repeats 1 --iterations 3

    # From JSON config
    EVERYROW_API_KEY=<your-api-key> uv run \
        python -m experiment_runner --config config.json

Required environment variables:
    EVERYROW_API_KEY - Your EveryRow API key (get from everyrow.io dashboard)
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dataset_config import DatasetConfig, get_dataset
from experiment_types import (
    ExperimentConfig,
    ExperimentResult,
    IterationResult,
    OracleLabelResult,
    OracleQueryResult,
    OracleType,
    RepeatResult,
)
from model import create_vectorizer, get_uncertainty_scores, train_classifier
from sklearn.model_selection import train_test_split
from utils import _get_balanced_seed_indices, parse_args

from everyrow.ops import agent_map
from everyrow.session import create_session
from everyrow.task import EffortLevel

# Configure logging with timestamps and flush
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)
# Force unbuffered stdout so logs appear in real time
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Not available in some environments (e.g. Jupyter)


async def query_llm_oracle(
    input_df: pd.DataFrame,
    dataset_config: DatasetConfig,
) -> OracleQueryResult:
    session_name = f"{dataset_config.name} Active Learning"
    log.info("  LLM oracle: creating session for %d samples...", len(input_df))
    async with create_session(name=session_name) as session:
        session_url = session.get_url()
        session_id_str = str(session.session_id)
        log.info(
            "  LLM oracle: session %s created, calling agent_map...", session_id_str
        )

        start_time = time.time()
        result = await agent_map(
            session=session,
            task=dataset_config.oracle_task,
            input=input_df,
            response_model=dataset_config.response_model,
            effort_level=EffortLevel.LOW,
        )
        latency_seconds = time.time() - start_time
        log.info("  LLM oracle: agent_map returned in %.1fs", latency_seconds)

        all_labels: list[OracleLabelResult] = []
        for i in range(len(input_df)):
            try:
                category = result.data["category"].iloc[i]
                label_id = dataset_config.category_to_id.get(category)
                all_labels.append(
                    OracleLabelResult(
                        label=label_id,
                        category=category,
                    )
                )
            except Exception as e:
                log.warning("  Failed to parse result for row %d: %s", i, e)
                all_labels.append(OracleLabelResult(label=None, category="unknown"))

    return OracleQueryResult(
        labels=all_labels,
        session_url=session_url,
        session_id=session_id_str,
        latency_seconds=latency_seconds,
    )


async def _run_active_learning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_config: DatasetConfig,
    oracle_type: OracleType,
    seed_size: int,
    query_size: int,
    iterations: int,
    seed_random_state: int,
    vectorizer,
) -> tuple[list[IterationResult], list[str]]:
    """Run active learning loop with given oracle.

    Args:
        train_df: Training data (unlabeled pool + will sample seed from here)
        test_df: Fixed test set for evaluation
        dataset_config: Dataset configuration
        oracle_type: GROUND_TRUTH or LLM
        seed_size: Number of initial labeled examples
        query_size: Number of examples to query per iteration
        iterations: Number of AL query rounds (iteration 0 = seed evaluation, no query)
        seed_random_state: Random state for seed sampling (varies across repeats)
        vectorizer: Pre-fitted TF-IDF vectorizer for consistent features

    Returns:
        Tuple of (history, session_ids)
    """
    oracle_tag = oracle_type.value.upper()
    rng = np.random.default_rng(seed_random_state)
    train_df = train_df.reset_index(drop=True)

    # Sample balanced seed dataset
    labeled_indices = _get_balanced_seed_indices(
        df=train_df,
        seed_size=seed_size,
        random_state=seed_random_state,
    )
    all_indices = set(range(len(train_df)))
    unlabeled_set = all_indices - set(labeled_indices)

    unlabeled_indices = list(unlabeled_set)
    rng.shuffle(unlabeled_indices)

    # Initialize labels dict with ground truth for seed
    labels: dict[int, int] = {
        idx: train_df.iloc[idx]["label"] for idx in labeled_indices
    }

    log.info(
        "  [%s] Seed: %d labeled, %d unlabeled",
        oracle_tag,
        len(labeled_indices),
        len(unlabeled_set),
    )

    history = []
    session_ids = []
    loop_start = time.time()

    for iteration in range(iterations + 1):
        iter_start = time.time()

        # Build labeled dataset
        labeled_df = train_df.iloc[labeled_indices][["text"]].copy()
        labeled_df["label"] = [labels[idx] for idx in labeled_indices]
        labeled_df = labeled_df.reset_index(drop=True)

        # Train and evaluate
        classifier, metrics = train_classifier(
            labeled_df, test_df, vectorizer=vectorizer
        )

        iteration_result = IterationResult(
            iteration=iteration,
            n_labeled=len(labeled_indices),
            accuracy=metrics["accuracy"],
            f1_macro=metrics["f1_macro"],
        )

        # Stop if this is the last iteration or no more unlabeled data
        if iteration == iterations or len(unlabeled_set) == 0:
            history.append(iteration_result)
            log.info(
                "  [%s] iter %d/%d: n_labeled=%d, acc=%.3f, f1=%.3f (final, %.1fs total)",
                oracle_tag,
                iteration,
                iterations,
                len(labeled_indices),
                metrics["accuracy"],
                metrics["f1_macro"],
                time.time() - loop_start,
            )
            break

        # Select most uncertain samples to query
        unlabeled_texts = train_df.iloc[unlabeled_indices]["text"].tolist()
        uncertainties = get_uncertainty_scores(classifier, unlabeled_texts)
        top_k_idx = np.argsort(uncertainties)[-query_size:]
        query_indices = [unlabeled_indices[i] for i in top_k_idx]

        # Query oracle
        if oracle_type == OracleType.GROUND_TRUTH:
            oracle_labels = train_df.iloc[query_indices]["label"].tolist()
            iteration_result.oracle_queried = len(query_indices)
            iteration_result.oracle_valid = len(query_indices)
            iteration_result.oracle_failures = 0
            iteration_result.oracle_accuracy = 1.0
        else:
            oracle_result = await query_llm_oracle(
                input_df=train_df.iloc[query_indices][["text"]],
                dataset_config=dataset_config,
            )
            oracle_labels = [r.label for r in oracle_result.labels]
            session_ids.append(oracle_result.session_id)

            # Calculate oracle accuracy vs ground truth
            true_labels = train_df.iloc[query_indices]["label"].tolist()
            oracle_correct = sum(
                o == t for o, t in zip(oracle_labels, true_labels) if o is not None
            )
            oracle_valid = sum(1 for o in oracle_labels if o is not None)
            oracle_failures = sum(1 for o in oracle_labels if o is None)

            iteration_result.oracle_queried = len(query_indices)
            iteration_result.oracle_valid = oracle_valid
            iteration_result.oracle_failures = oracle_failures
            iteration_result.oracle_accuracy = oracle_correct / max(oracle_valid, 1)
            iteration_result.session_url = oracle_result.session_url
            iteration_result.session_id = oracle_result.session_id
            iteration_result.query_latency_seconds = oracle_result.latency_seconds

            if oracle_failures > 0:
                log.warning(
                    "    %d/%d oracle failures", oracle_failures, len(query_indices)
                )

        iter_elapsed = time.time() - iter_start
        oracle_acc_str = (
            f", oracle_acc={iteration_result.oracle_accuracy:.1%}"
            if iteration_result.oracle_accuracy is not None
            else ""
        )
        log.info(
            "  [%s] iter %d/%d: n_labeled=%d, acc=%.3f, f1=%.3f%s (%.1fs)",
            oracle_tag,
            iteration,
            iterations,
            len(labeled_indices),
            metrics["accuracy"],
            metrics["f1_macro"],
            oracle_acc_str,
            iter_elapsed,
        )

        history.append(iteration_result)

        # Update labeled set with oracle responses
        for idx, label in zip(query_indices, oracle_labels):
            unlabeled_set.discard(idx)
            if label is not None:
                labeled_indices.append(idx)
                labels[idx] = label

        unlabeled_indices = [i for i in unlabeled_indices if i in unlabeled_set]

    return history, session_ids


def _build_result(
    config: ExperimentConfig,
    repeat_results: list[RepeatResult],
    start_time: datetime,
    experiment_id: str | None = None,
) -> ExperimentResult:
    """Build an ExperimentResult from accumulated repeat results."""
    gt_accs = [r.ground_truth_final_accuracy for r in repeat_results]
    llm_accs = [r.llm_final_accuracy for r in repeat_results]
    oracle_accs = [r.llm_oracle_accuracy for r in repeat_results]

    kwargs = {}
    if experiment_id is not None:
        kwargs["experiment_id"] = experiment_id

    return ExperimentResult(
        config=config,
        repeat_results=repeat_results,
        repeats_completed=len(repeat_results),
        mean_ground_truth_accuracy=float(np.mean(gt_accs)),
        std_ground_truth_accuracy=float(np.std(gt_accs)),
        mean_llm_accuracy=float(np.mean(llm_accs)),
        std_llm_accuracy=float(np.std(llm_accs)),
        mean_llm_oracle_accuracy=float(np.mean(oracle_accs)),
        total_duration_seconds=(datetime.now() - start_time).total_seconds(),
        **kwargs,
    )


def _save_result(result: ExperimentResult, output_path) -> None:
    """Save experiment result to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result.model_dump(), f, indent=2, default=str)


async def run_experiment(
    config: ExperimentConfig,
    existing_result: ExperimentResult | None = None,
    output_path=None,
) -> ExperimentResult:
    """Run the full experiment with multiple repeats.

    Each repeat uses a different random seed for the initial labeled set,
    but the same fixed train/test split. This isolates the effect of
    seed selection from test set variation.
    """
    start_time = datetime.now()

    dataset_config = get_dataset(config.dataset)
    full_df = dataset_config.load(random_state=config.random_state)

    # Create fixed train/test split (stratified)
    train_df, test_df = train_test_split(
        full_df,
        test_size=config.test_fraction,
        stratify=full_df["label"],
        random_state=config.random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    log.info("Fixed split: train=%d, test=%d", len(train_df), len(test_df))

    # Create vectorizer on training data only to avoid test leakage
    vectorizer = create_vectorizer(train_df["text"].tolist())
    log.info(
        "Vectorizer fitted on %d texts, %d features",
        len(train_df),
        len(vectorizer.vocabulary_),
    )

    # Resume from existing results if provided
    if existing_result:
        repeat_results = list(existing_result.repeat_results)
        start_repeat = existing_result.repeats_completed
        experiment_id = existing_result.experiment_id
        log.info(
            "Resuming from repeat %d (already completed %d)",
            start_repeat + 1,
            start_repeat,
        )
    else:
        repeat_results = []
        start_repeat = 0
        experiment_id = None

    for repeat_idx in range(start_repeat, config.repeats):
        # Each repeat uses a different seed for the initial labeled set
        seed_random_state = config.random_state + repeat_idx + 1
        repeat_start = time.time()

        log.info(
            "--- Repeat %d/%d (seed_state=%d) ---",
            repeat_idx + 1,
            config.repeats,
            seed_random_state,
        )

        # Run GT oracle
        log.info("  [GT] Starting...")
        gt_history, _ = await _run_active_learning(
            train_df=train_df,
            test_df=test_df,
            dataset_config=dataset_config,
            oracle_type=OracleType.GROUND_TRUTH,
            seed_size=config.seed_size,
            query_size=config.query_size,
            iterations=config.iterations,
            seed_random_state=seed_random_state,
            vectorizer=vectorizer,
        )

        # Run LLM oracle with same seed
        log.info("  [LLM] Starting...")
        llm_history, session_ids = await _run_active_learning(
            train_df=train_df,
            test_df=test_df,
            dataset_config=dataset_config,
            oracle_type=OracleType.LLM,
            seed_size=config.seed_size,
            query_size=config.query_size,
            iterations=config.iterations,
            seed_random_state=seed_random_state,
            vectorizer=vectorizer,
        )

        # Calculate average oracle accuracy for this repeat
        llm_oracle_accs = [
            h.oracle_accuracy for h in llm_history if h.oracle_accuracy is not None
        ]
        avg_oracle_acc = (
            sum(llm_oracle_accs) / len(llm_oracle_accs) if llm_oracle_accs else 0.0
        )

        repeat_elapsed = time.time() - repeat_start
        log.info(
            "  Repeat %d/%d done in %.1fs — GT: %.3f, LLM: %.3f, oracle_acc: %.1f%%",
            repeat_idx + 1,
            config.repeats,
            repeat_elapsed,
            gt_history[-1].accuracy,
            llm_history[-1].accuracy,
            avg_oracle_acc * 100,
        )

        repeat_results.append(
            RepeatResult(
                repeat=repeat_idx,
                seed_random_state=seed_random_state,
                train_size=len(train_df),
                test_size=len(test_df),
                ground_truth_history=gt_history,
                llm_history=llm_history,
                ground_truth_final_accuracy=gt_history[-1].accuracy,
                ground_truth_final_f1=gt_history[-1].f1_macro,
                llm_final_accuracy=llm_history[-1].accuracy,
                llm_final_f1=llm_history[-1].f1_macro,
                llm_oracle_accuracy=avg_oracle_acc,
                session_ids=session_ids,
            )
        )

        # Save after each repeat so partial results are preserved
        if output_path:
            partial_result = _build_result(
                config, repeat_results, start_time, experiment_id
            )
            # Preserve the experiment_id from the first save onwards
            experiment_id = partial_result.experiment_id
            _save_result(partial_result, output_path)
            log.info(
                "  Saved partial result (%d/%d repeats) to %s",
                len(repeat_results),
                config.repeats,
                output_path,
            )

    return _build_result(config, repeat_results, start_time, experiment_id)


async def async_main():
    args = parse_args()

    existing_result = None

    if args.resume:
        # Load existing experiment to resume
        log.info("Loading existing experiment from %s", args.resume)
        existing_data = json.loads(args.resume.read_text())
        existing_result = ExperimentResult.model_validate(existing_data)

        # Use config from existing experiment, but override repeats with CLI arg
        old_config = existing_result.config
        config = ExperimentConfig(
            dataset=old_config.dataset,
            test_fraction=old_config.test_fraction,
            repeats=args.repeats,  # Always use CLI arg for total repeats
            seed_size=old_config.seed_size,
            query_size=old_config.query_size,
            iterations=old_config.iterations,
            random_state=old_config.random_state,
            version=old_config.version,
        )
        log.info(
            "Target: %d total repeats, %d already done",
            config.repeats,
            existing_result.repeats_completed,
        )

    elif args.config:
        config = ExperimentConfig.model_validate_json(args.config.read_text())
    else:
        config_dict = {
            "dataset": args.dataset,
            "test_fraction": args.test_fraction,
            "repeats": args.repeats,
            "seed_size": args.seed_size,
            "query_size": args.query_size,
            "iterations": args.iterations,
            "random_state": args.random_state,
            "version": args.version,
        }
        config = ExperimentConfig.model_validate(config_dict)

    log.info(
        "Config: dataset=%s, repeats=%d, seed=%d, query=%d, iters=%d, version=%s",
        config.dataset,
        config.repeats,
        config.seed_size,
        config.query_size,
        config.iterations,
        config.version or "(none)",
    )

    # Determine output path before running so we can save incrementally
    if args.resume:
        output_path = args.resume
    else:
        postfix = f"v{config.version}_" if config.version else ""
        postfix += f"{config.dataset}_{config.repeats}repeats"
        fname = f"experiment_{postfix}.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / fname

    result = await run_experiment(
        config, existing_result=existing_result, output_path=output_path
    )

    # Final save (redundant but explicit)
    _save_result(result, output_path)
    log.info("Saved final result to %s", output_path)

    # Print summary
    log.info("=" * 50)
    log.info(
        "GT Accuracy:  %.3f ± %.3f",
        result.mean_ground_truth_accuracy,
        result.std_ground_truth_accuracy,
    )
    log.info(
        "LLM Accuracy: %.3f ± %.3f", result.mean_llm_accuracy, result.std_llm_accuracy
    )
    log.info("LLM Oracle:   %.1f%%", result.mean_llm_oracle_accuracy * 100)
    log.info("Duration:     %.1fs", result.total_duration_seconds)


if __name__ == "__main__":
    asyncio.run(async_main())
