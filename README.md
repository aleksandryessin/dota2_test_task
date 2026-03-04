# Dota 2 First-Pick Hero Prediction

Predict which hero will be **first-picked** in professional Dota 2 Captain's Mode matches.

**Metric**: Accuracy@5 — fraction of matches where the true first-picked hero is in the model's top-5 predictions.

## Best Result

| | |
|---|---|
| **Model** | LightGBM with V2 features |
| **Acc@5** | **0.475** (95/200) |
| **Acc@1** | **0.225** (45/200) |
| **Training** | 300 rounds, ~7 min |

## All Models Comparison

| Model | Acc@1 | Acc@3 | Acc@5 | Notes |
|---|:---:|:---:|:---:|---|
| Popularity | 0.010 | 0.070 | 0.110 | Frequency baseline |
| LightGBM V1 | 0.015 | 0.110 | 0.125 | Basic features |
| LSTM (bidir) | 0.085 | 0.190 | 0.260 | Ban sequence only |
| Transformer V1 (default) | 0.080 | 0.225 | 0.320 | Ban + context |
| Transformer V1 (Optuna) | 0.090 | 0.230 | 0.405 | 50 HPO trials |
| CandidateScorerNet V2 | 0.150 | 0.325 | 0.425 | + priors, focal, series |
| **LightGBM V2** | **0.225** | **0.360** | **0.475** | **V2 features** |
| Ensemble (neural + GBM) | 0.225 | 0.360 | 0.475 | = LightGBM alone |

## V2 Feature Engineering

Key innovations that drove the improvement from 0.125 → 0.475:

- **Captain/team target encoding**: smoothed P(hero | captain/team) — strongest signal
- **Rolling meta**: per-sample hero pick frequency over last 5000 matches
- **Series context**: game position + prior first-picks within series
- **League ID**: normalized tournament identifier
- **Ban multi-hot**: binary hero-in-ban encoding (LightGBM)
- **Ban masking**: banned heroes excluded from predictions at inference

## V2 CandidateScorerNet Architecture

```
Ban sequence (6 tokens)
    ↓
Embedding (128-dim, no positional encoding)
    ↓
TransformerEncoder (2 layers, 2 heads, GELU)
    ↓ mean pool
    ├── Context (9-dim) → Linear → ctx_repr
    ↓
[ban_repr, ctx_repr] → MLP → match_repr (128-dim)
    ↓
dot(match_repr, hero_embeddings) + w·log(priors) + bias
    ↓
Logits → Ban-masked Top-5
```

Prior channels: captain distribution, team distribution, rolling meta, series history.

## Dataset

| | |
|---|---|
| **Matches** | 92,685 (2023-01-01 to 2026-01-31) |
| **Train** | 92,485 (all but last 200) |
| **Val** | 200 (last 200, temporal split) |
| **Heroes** | 127 unique |

## Project Structure

```
├── notebooks/
│   ├── full_pipeline.ipynb       # V1 pipeline (Transformer, Acc@5=0.405)
│   └── v2_pipeline.ipynb         # V2 pipeline (CandidateScorer, Acc@5=0.425)
├── src/
│   ├── data/
│   │   ├── features_v2.py        # V2 feature engineering
│   │   └── loader.py             # Data loading
│   ├── models/
│   │   ├── candidate_scorer.py   # CandidateScorerNet + CrossAttentionScorerNet
│   │   └── transformer_model.py  # Transformer V1
│   ├── evaluation/               # Metrics
│   └── tracking/                 # MLflow integration
├── scripts/
│   ├── run_baselines.py          # All baseline models
│   ├── run_transformers.py       # Optuna HPO for Transformer V1
│   ├── run_v2.py                 # V2 CandidateScorerNet pipeline
│   ├── run_lgbm_v2.py            # LightGBM with V2 features
│   └── run_ensemble.py           # Ensemble neural + LightGBM
├── configs/default.yaml
├── data/test_task_dataset.csv    # DVC-tracked
├── mlflow.db                     # Experiment tracking
├── Strategy.md
└── Workflow.md
```

## Quick Start

```bash
# Setup
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Restore data
dvc pull

# Run baselines
python scripts/run_baselines.py

# Run best model (LightGBM V2)
python scripts/run_lgbm_v2.py

# Run V2 neural net
python scripts/run_v2.py

# Run ensemble
python scripts/run_ensemble.py

# View all results in MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Reproduce Best Result

- **LightGBM V2 (0.475)**: `python scripts/run_lgbm_v2.py`
- **Neural V2 (0.425)**: open `notebooks/v2_pipeline.ipynb`
- **Transformer V1 (0.405)**: open `notebooks/full_pipeline.ipynb`

## Notes

- Neural models run on **MPS** (Apple Silicon GPU) when available
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set for Transformer compatibility
- LightGBM is CPU-only (OpenMP threading)

## Tools

- **uv** — fast dependency management
- **MLflow** — experiment tracking
- **Optuna** — Bayesian HPO
- **DVC** — data versioning
- **LightGBM** — gradient boosting
- **PyTorch** — neural networks
