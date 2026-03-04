# Workflow

## Completed

- [x] Project setup: Python 3.11 via uv, all dependencies installed
- [x] Project restructured from notebook → `src/` packages
- [x] Data pipeline: loading, draft parsing, feature engineering
- [x] Popularity baseline → Acc@5: 0.110
- [x] LightGBM baseline (multiclass on ban + time + context) → Acc@5: 0.125
- [x] LSTM baseline (bidirectional, ban sequence + context, MPS GPU) → Acc@5: 0.260
- [x] Transformer baseline (self-attention on ban sequence, MPS GPU) → Acc@5: 0.320
- [x] MLflow integration — all runs logged
- [x] DVC initialization for data versioning
- [x] Documentation: README, Strategy, Workflow
- [x] MPS (Apple Silicon GPU) support with OpenMP/LightGBM compatibility fix
- [x] Optuna HPO for Transformer: 50 trials, MAX_EPOCHS=100, PATIENCE=10
- [x] Best result: **Acc@5 = 0.405** (trial #037, embed=128, nhead=2, layers=3, head_dim=512)
- [x] Full pipeline notebook: `notebooks/full_pipeline.ipynb`
- [x] README updated with results table and architecture diagram

## Current Step

- [ ] Decide on next improvement direction (feature engineering vs architecture)

## Next Steps

- [ ] Add captain/team historical features (target encoding)
- [ ] Add rolling meta features (hero popularity in recent window)
- [ ] Try candidate-scoring architecture (score per hero instead of 127-class softmax)
- [ ] Data augmentation (radiant/dire side swap)
- [ ] Increase val_size for more stable evaluation
- [ ] Build ensemble of best models (LightGBM + Transformer stacking)
