# Dota 2 First-Pick Hero Prediction

## Problem

Predict which hero will be **first-picked** in professional Dota 2 Captain's Mode matches.

**Metric**: Accuracy@5 — fraction of matches where the true first-picked hero is in the model's top-5 predictions.

## Dataset

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

**92,685 matches** (2023-01-01 → 2026-01-31). Train: 92,485 / Val: last 200 (temporal split).

## Features Available at Prediction Time

Before the first pick, we observe:
1. **Pre-first-pick bans** — 4 bans (2 per team) at draft orders 0–3
2. **First-pick team** — which side (radiant/dire) picks first
3. **Team & captain IDs** — encode historical drafting preferences
4. **Match context** — league, cluster, series type
5. **Time features** — meta shifts, patch eras

## Preprocessing Pipeline

1. Parse `picks_bans` JSON → extract pre-first-pick bans + first-pick team
2. Build hero vocabulary (all heroes across all drafts)
3. Build target vocabulary (heroes seen as first-picks in training)
4. Feature engineering:
   - **Tabular** (LightGBM): multi-hot ban vector + time + context
   - **Sequential** (LSTM/Transformer): ordered ban embeddings + context vector

## Models

| Model | Approach |
|---|---|
| **Popularity** | Rank by global first-pick frequency |
| **LightGBM** | Multiclass gradient boosting on tabular features |
| **LSTM** | Bidirectional LSTM over ban sequence + context |
| **Transformer** | Self-attention over ban sequence + context |

## Project Structure

```
├── src/
│   ├── data/           # Loading, parsing, feature engineering
│   ├── models/         # Model implementations (popularity, lgbm, lstm, transformer)
│   ├── evaluation/     # Metrics (accuracy@k)
│   └── tracking/       # MLflow integration
├── scripts/
│   └── run_baselines.py
├── configs/
│   └── default.yaml    # All hyperparameters
├── data/
│   └── test_task_dataset.csv (DVC-tracked)
├── mlruns/             # MLflow tracking store (gitignored)
├── pyproject.toml
├── requirements.txt    # Pinned dependencies
├── Strategy.md
└── Workflow.md
```

## Baseline Results

| Model | Acc@1 | Acc@3 | Acc@5 | Time |
|---|---|---|---|---|
| Popularity | 0.010 | 0.070 | 0.110 | 0.0s |
| LightGBM | 0.015 | 0.110 | 0.125 | 195s |
| LSTM (MPS) | 0.085 | 0.190 | 0.260 | 12s |
| **Transformer (MPS)** | **0.080** | **0.225** | **0.320** | 22s |

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

# View results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://127.0.0.1:5000
```

## Notes

- Neural models (LSTM, Transformer) run on **MPS** (Apple Silicon GPU)
- Due to OpenMP/MPS conflict, neural models train **before** LightGBM
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set for Transformer compatibility

## Tools

- **uv** — fast dependency management with version locking
- **MLflow** — experiment tracking, metric comparison, model registry
- **DVC** — data versioning (tracks `data/test_task_dataset.csv`)
