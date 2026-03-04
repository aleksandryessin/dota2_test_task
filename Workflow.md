# Workflow

## Completed

- [x] Project setup: Python 3.11 via uv, all dependencies installed
- [x] Project restructured from notebook → `src/` packages
- [x] Data pipeline: loading, draft parsing, feature engineering
- [x] Popularity baseline → Acc@5: 0.110
- [x] LightGBM baseline (multiclass on ban + time + context) → Acc@5: 0.125
- [x] LSTM baseline (bidirectional, ban sequence + context, MPS GPU) → Acc@5: 0.260
- [x] Transformer baseline (self-attention on ban sequence, MPS GPU) → Acc@5: **0.320**
- [x] MLflow integration — all 4 runs logged
- [x] DVC initialization for data versioning
- [x] Documentation: README, Strategy, Workflow
- [x] MPS (Apple Silicon GPU) support with OpenMP/LightGBM compatibility fix

## Current Step

- [ ] Analyze results: error analysis, which heroes are hardest to predict
- [ ] Review MLflow UI, compare runs visually

## Next Steps

- [ ] Add captain/team historical features (target encoding)
- [ ] Add rolling meta features (hero popularity in recent window)
- [ ] Tune LightGBM with Optuna (features matter more than tree params here)
- [ ] Increase Transformer epochs / enlarge model (still improving at epoch 20)
- [ ] Try data augmentation (radiant/dire side swap)
- [ ] Build ensemble of best models
