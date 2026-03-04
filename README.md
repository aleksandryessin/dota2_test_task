# Dota 2 First-Pick Hero Prediction

Predict which hero will be **first-picked** in professional Dota 2 Captain's Mode matches.

**Metric**: Accuracy@5 — fraction of matches where the true first-picked hero is in the model's top-5 predictions.

## Best Result

| | |
|---|---|
| **Model** | Transformer Encoder + MLP head |
| **Acc@5** | **0.405** (81/200) |
| **Training** | 50 epochs, 2.5 min on Apple M-series (MPS) |

Best configuration (Optuna trial #037, extended to 100 epochs):

| Parameter | Value |
|---|---|
| `embed_dim` | 128 |
| `nhead` | 2 |
| `num_layers` | 3 |
| `ff_mult` | 4 (feedforward = 512) |
| `head_dim` | 512 |
| `dropout` | 0.439 |
| `lr` | 9.74e-4 |
| `weight_decay` | 2.50e-5 |
| `batch_size` | 512 |
| `label_smoothing` | 0.073 |

## Baseline Comparison

| Model | Acc@1 | Acc@3 | Acc@5 | Time |
|---|:---:|:---:|:---:|:---:|
| Popularity | 0.010 | 0.070 | 0.110 | <1s |
| LightGBM | 0.015 | 0.110 | 0.125 | 195s |
| LSTM (bidir) | 0.085 | 0.190 | 0.260 | 12s |
| Transformer (default) | 0.080 | 0.225 | 0.320 | 22s |
| **Transformer (tuned)** | — | — | **0.405** | 155s |

## Optuna Hyperparameter Search (50 trials)

Selected trials showing the effect of architecture choices:

| Trial | Acc@5 | embed | heads | layers | head_dim | dropout | lr | batch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **#037** | **0.395** | 128 | 2 | 3 | 512 | 0.439 | 9.7e-4 | 512 |
| #039 | 0.395 | 128 | 2 | 1 | 512 | 0.396 | 4.9e-3 | 512 |
| #011 | 0.375 | 64 | 4 | 3 | 512 | 0.333 | 1.6e-3 | 256 |
| #031 | 0.375 | 128 | 2 | 3 | 512 | 0.319 | 1.9e-3 | 1024 |
| #047 | 0.370 | 128 | 4 | 2 | 512 | 0.321 | 1.6e-3 | 512 |
| #024 | 0.365 | 64 | 8 | 3 | 512 | 0.245 | 6.2e-4 | 1024 |
| #014 | 0.355 | 32 | 8 | 4 | 256 | 0.242 | 2.0e-3 | 256 |
| #004 | 0.340 | 32 | 8 | 3 | 512 | 0.303 | 1.7e-3 | 256 |
| #001 | 0.320 | 32 | 8 | 1 | 128 | 0.477 | 4.3e-3 | 1024 |
| #010 | 0.255 | 32 | 4 | 3 | 512 | 0.067 | 5.4e-5 | 256 |

Key patterns: `embed_dim=128` and `head_dim=512` consistently dominate; 2-3 encoder layers are optimal; dropout ~0.3-0.45 with moderate label smoothing ~0.07-0.09.

## Dataset

| | |
|---|---|
| **Matches** | 92,685 (2023-01-01 to 2026-01-31) |
| **Train** | 92,485 (all but last 200) |
| **Val** | 200 (last 200, temporal split) |
| **Heroes** | 127 unique |
| **Target classes** | 127 (heroes seen as first-picks) |

| Column | Description |
|---|---|
| `match_id` | Unique match identifier |
| `start_time` | Unix timestamp (seconds) |
| `radiant_win` | Post-match outcome (not usable for prediction) |
| `leagueid` | Tournament/league ID |
| `cluster` | Server region |
| `radiant/dire_team_id` | Registered team IDs |
| `radiant/dire_captain` | Captain (drafter) account IDs |
| `series_id`, `series_type` | Series context (Bo1/Bo3/Bo5) |
| `picks_bans` | Full 24-action draft sequence (JSON) |
| `first_pick_hero` | **Target** — hero_id of the first picked hero |

## Architecture

```
Ban sequence (6 tokens)
    ↓
Embedding (128-dim) + Positional Embedding
    ↓
TransformerEncoder (3 layers, 2 heads, GELU, dim_ff=512)
    ↓
LayerNorm → Mean pooling (masked)
    ↓
Concatenate with context vector (6-dim)
    ↓
MLP head: Linear(134→512) → GELU → Dropout
        → Linear(512→256) → GELU → Dropout
        → Linear(256→127)
    ↓
Softmax → Top-5 predictions
```

**Input features** (available before first pick):
- **Ban sequence**: up to 6 hero IDs banned before the first pick, encoded as padded integer sequence
- **Context vector** (6-dim): first-pick team side, year, month, day-of-week, hour, server cluster

## Project Structure

```
├── notebooks/
│   └── full_pipeline.ipynb    # End-to-end reproducible pipeline
├── src/
│   ├── data/                  # Loading, parsing, feature engineering
│   ├── models/                # Model implementations
│   ├── evaluation/            # Metrics (accuracy@k)
│   └── tracking/              # MLflow integration
├── scripts/
│   ├── run_baselines.py       # Train all baseline models
│   ├── run_transformers.py    # Optuna HPO for Transformer
│   └── run_single_trial.py   # Single trial with fixed params
├── configs/
│   └── default.yaml           # Hyperparameters
├── data/
│   └── test_task_dataset.csv  # DVC-tracked dataset
├── first_pick_task.ipynb      # Original task description
├── mlflow.db                  # MLflow tracking store
├── Strategy.md
└── Workflow.md
```

## Quick Start

```bash
# Setup
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Restore data (if cloned fresh)
dvc pull

# Run all baselines
python scripts/run_baselines.py

# Run Optuna HPO (50 trials)
python scripts/run_transformers.py

# Run best single trial (100 epochs)
python scripts/run_single_trial.py

# View results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

## Reproduce Best Result

Open `notebooks/full_pipeline.ipynb` — it contains the complete pipeline from data loading to the 0.405 result in a single notebook.

## Notes

- Neural models run on **MPS** (Apple Silicon GPU) when available, falls back to CPU
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set for Transformer compatibility
- Due to OpenMP/MPS conflict, neural models must train **before** LightGBM in `run_baselines.py`

## Tools

- **uv** — fast dependency management with version locking
- **MLflow** — experiment tracking, metric comparison
- **Optuna** — Bayesian hyperparameter optimization (TPE sampler)
- **DVC** — data versioning (tracks `data/test_task_dataset.csv`)
