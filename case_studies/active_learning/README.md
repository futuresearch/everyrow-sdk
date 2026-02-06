# Active Learning with an LLM Oracle

Use an LLM as a labeling oracle in an active learning pipeline for text classification, using the [EveryRow SDK](https://github.com/futuresearch/everyrow-sdk).

Active learning iteratively selects the most informative samples to be labeled, rather than labeling all data upfront. At each iteration the model identifies examples it is most uncertain about and sends them to an **oracle** for labeling. Traditionally the oracle is a human annotator (or a group of them), making the process expensive and slow. Here we replace the human with an LLM via `everyrow.agent_map`, enabling fast, scalable, and cost-effective labeling.

## Overview

We use TF-IDF + LightGBM as the classifier on the [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) text classification task (14 categories). The pipeline:

1. **Seed** with a small balanced set of 700 ground-truth-labeled examples (50 per class)
2. **Train** a TF-IDF + LightGBM classifier on the labeled data
3. **Select** the 20 most uncertain samples from the unlabeled pool (entropy-based uncertainty sampling)
4. **Label** selected samples using `everyrow.agent_map` (the LLM oracle)
5. **Repeat** for 10 iterations, adding LLM-labeled samples to the training set each round

The experiment runner also runs a parallel ground-truth oracle as a baseline, and repeats with different random seeds to measure variance.

**Cost:** Each full run (10 iterations, 200 annotations) costs ~$0.26. See [everyrow.io](https://everyrow.io) for docs and pricing ($20 free credit).

## Getting Started

### 1. Get an API Key

Get your key from [everyrow.io/api-key](https://everyrow.io/api-key) ($20 free credit).

### 2. Install Dependencies

```bash
cd case_studies/active_learning
uv sync
```

### 3. Try the Notebook

The easiest way to get started is the interactive tutorial notebook:

```bash
export EVERYROW_API_KEY=<your-api-key>
uv run python -m ipykernel install --user --name active-learning-tutorial
```

Then open `active_learning_tutorial.ipynb` in VS Code or JupyterLab and select the **active-learning-tutorial** kernel.

The notebook also works on **Kaggle** and **Colab** — see the setup cells for instructions on configuring your API key in those environments.

### 4. Run the CLI Experiment Runner

For more rigorous experiments with multiple repeats and automated result saving:

```bash
export EVERYROW_API_KEY=<your-api-key>

# Quick test (1 repeat, 3 iterations)
uv run python -m experiment_runner \
    --repeats 1 \
    --iterations 3

# Full experiment (5 repeats, 10 iterations)
uv run python -m experiment_runner \
    --repeats 5 \
    --seed-size 700 \
    --query-size 20 \
    --iterations 10
```

### 5. View Results

```bash
uv run python -m view_results \
    --results-dir ./results \
    --show
```

## Example Results

From experiments on DBpedia-14 (5% stratified sample, 5 repeats):

| Oracle | Final Accuracy | Final F1 |
|--------|----------------|----------|
| Ground Truth | 79.4% | 79.5% |
| LLM | 79.7% | 79.6% |

**LLM Oracle Label Accuracy**: ~97% agreement with ground truth

Key findings:
- LLM oracle achieves comparable performance to ground truth labels
- High label accuracy (~97%) translates to nearly identical downstream classifier performance
- Active learning with LLM labels is a practical alternative to manual annotation

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
| `--resume` | `None` | Resume from an existing experiment file |

### JSON Config

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

### Resuming Experiments

If an experiment is interrupted, resume it:

```bash
uv run python -m experiment_runner \
    --resume ./results/experiment_v1_dbpedia_tiny_5repeats.json \
    --repeats 10  # Continue to 10 total repeats
```

## Dataset

The default dataset is [DBpedia-14](https://huggingface.co/datasets/fancyzhx/dbpedia_14), a text classification dataset with 14 categories: Company, Educational Institution, Artist, Athlete, Office Holder, Mean Of Transportation, Building, Natural Place, Village, Animal, Plant, Album, Film, Written Work.

- `dbpedia`: Full dataset (~560k train samples)
- `dbpedia_tiny`: 5% stratified sample (~28k samples) — recommended for testing

## Output

Results are saved as JSON in `./results/`:

```
experiment_v{version}_{dataset}_{repeats}repeats.json
```

Running `view_results` generates learning curve plots (accuracy, F1, oracle accuracy) and a CSV for custom analysis.

## Structure

```
active_learning/
├── README.md
├── pyproject.toml                    # Dependencies (uv sync to install)
├── active_learning_tutorial.ipynb    # Interactive tutorial notebook
├── experiment_runner.py              # Main experiment logic and CLI
├── dataset_config.py                 # Dataset configurations and loading
├── model.py                          # TextClassifier (TF-IDF + LightGBM)
├── experiment_types.py               # Pydantic models for configs and results
├── utils.py                          # CLI parsing and utility functions
└── view_results.py                   # Result visualization and plotting
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `EVERYROW_API_KEY` | Yes | Your EveryRow API key ([get one here](https://everyrow.io/api-key)) |
| `EVERYROW_API_URL` | No | API URL (defaults to production) |

## Links

- [EveryRow SDK on GitHub](https://github.com/futuresearch/everyrow-sdk)
- [EveryRow docs](https://everyrow.io)
- [DBpedia-14 dataset on HuggingFace](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
