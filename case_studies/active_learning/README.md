# Active Learning Experiments with EveryRow

This module runs active learning experiments comparing **LLM oracles** vs **ground truth oracles** for text classification tasks using the [EveryRow SDK](https://github.com/futuresearch/everyrow-sdk) ([docs](https://everyrow.io)).

## Overview

Active learning is a machine learning technique where the model iteratively selects the most informative samples to label, reducing the total labeling effort needed to achieve good performance.

This experiment framework compares two labeling strategies:
- **Ground Truth Oracle**: Uses the dataset's actual labels (baseline)
- **LLM Oracle**: Uses an LLM via EveryRow to generate labels

The key question: *Can an LLM oracle achieve comparable results to ground truth labels while being more practical for real-world scenarios?*

## How It Works

1. **Initial Seed**: Start with a small balanced set of labeled examples
2. **Train Classifier**: Train a TF-IDF + LightGBM classifier on labeled data
3. **Uncertainty Sampling**: Select the most uncertain samples from the unlabeled pool
4. **Query Oracle**: Get labels from either ground truth or LLM
5. **Repeat**: Add new labels to training set and iterate

The experiment runs multiple repeats with different random seeds to measure variance.

## Installation

```bash
cd case_studies/active_learning
uv sync  # installs all dependencies
```

## Quick Start

### 1. Get an API Key

Sign up at [everyrow.io](https://everyrow.io) and create an API key from your dashboard.

### 2. Run an Experiment

```bash
# Set your API key
export EVERYROW_API_KEY=<your-api-key>

# Run a quick test (1 repeat, 3 iterations)
uv run python -m experiment_runner \
    --repeats 1 \
    --iterations 3

# Run a full experiment (5 repeats, 10 iterations)
uv run python -m experiment_runner \
    --repeats 5 \
    --seed-size 700 \
    --query-size 20 \
    --iterations 10
```

### 3. View Results

```bash
uv run python -m view_results \
    --results-dir ./results \
    --show
```

## Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `dbpedia_tiny` | Dataset to use (`dbpedia` or `dbpedia_tiny`) |
| `--repeats` | `5` | Number of experiment repeats with different seeds |
| `--seed-size` | `700` | Initial labeled examples (balanced across classes) |
| `--query-size` | `20` | Samples to query per iteration |
| `--iterations` | `10` | Number of active learning iterations |
| `--test-fraction` | `0.2` | Fraction held out for testing |
| `--random-state` | `42` | Base random seed for reproducibility |
| `--version` | `""` | Version tag for organizing results |
| `--output-dir` | `./results` | Directory to save results |
| `--resume` | `None` | Resume from existing experiment file |

### JSON Config File

You can also use a JSON config file:

```json
{
  "dataset": "dbpedia_tiny",
  "repeats": 5,
  "seed_size": 700,
  "query_size": 20,
  "iterations": 10,
  "test_fraction": 0.2,
  "random_state": 42,
  "version": "my_experiment"
}
```

```bash
uv run python -m experiment_runner --config config.json
```

## Datasets

### DBpedia-14

The default dataset is [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14), a text classification dataset with 14 categories:

| Category | Description |
|----------|-------------|
| Company | Business organizations |
| Educational Institution | Schools, universities |
| Artist | Musicians, painters, actors |
| Athlete | Sports players |
| Office Holder | Politicians, officials |
| Mean Of Transportation | Vehicles, aircraft, ships |
| Building | Structures, landmarks |
| Natural Place | Geographic features |
| Village | Small settlements |
| Animal | Animal species |
| Plant | Plant species |
| Album | Music albums |
| Film | Movies |
| Written Work | Books, articles |

- `dbpedia`: Full dataset (~560k train samples)
- `dbpedia_tiny`: 5% stratified sample (~28k samples) - recommended for testing

## Output

### Results JSON

Each experiment produces a JSON file in `./results/`:

```
experiment_v{version}_{dataset}_{repeats}repeats.json
```

Contains:
- Configuration parameters
- Per-repeat results (accuracy, F1, oracle accuracy at each iteration)
- Aggregate statistics (mean/std across repeats)

### Plots

Running `view_results` generates:

| Plot | Description |
|------|-------------|
| `learning_curves.png` | Accuracy and F1 vs labeled examples |
| `learning_curve_accuracy.png` | Accuracy comparison with summary stats |
| `oracle_accuracy.png` | LLM oracle accuracy over iterations |
| `results.csv` | Full results as CSV for custom analysis |

## Structure

```
active_learning/
├── README.md
├── pyproject.toml                # Dependencies (uv sync to install)
├── experiment_runner.py          # Main experiment logic and CLI
├── dataset_config.py             # Dataset configurations and loading
├── model.py                      # TextClassifier (TF-IDF + LightGBM)
├── types.py                      # Pydantic models for configs and results
├── utils.py                      # CLI parsing and utility functions
└── view_results.py               # Result visualization and plotting
```

## Example Results

From experiments on DBpedia-14 (tiny):

| Oracle | Final Accuracy | Final F1 |
|--------|----------------|----------|
| Ground Truth | 79.4% | 79.5% |
| LLM | 79.7% | 79.6% |

**LLM Oracle Label Accuracy**: ~97% agreement with ground truth

Key findings:
- LLM oracle achieves comparable performance to ground truth
- High label accuracy (~97%) translates to nearly identical classifier performance
- Active learning with LLM labels is a viable alternative to manual labeling

## Resuming Experiments

If an experiment is interrupted, you can resume it:

```bash
uv run python -m experiment_runner \
    --resume ./results/experiment_v1_dbpedia_tiny_5repeats.json \
    --repeats 10  # Continue to 10 total repeats
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EVERYROW_API_KEY` | Yes | Your EveryRow API key |
| `EVERYROW_API_URL` | No | API URL (defaults to production) |

## License

See repository LICENSE file.
