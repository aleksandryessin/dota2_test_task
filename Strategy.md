# Strategy

## Current State

Four baselines established: Popularity, LightGBM, LSTM, Transformer.
All logged to MLflow for comparison.

## Improvement Axes

### 1. Feature Engineering (highest expected impact)

- **Captain/team hero pools**: historical first-pick distribution per captain/team (target encoding with smoothing)
- **Ban-context features**: which hero archetypes were banned (carry, mid, support), ban synergy patterns
- **Temporal meta features**: rolling hero pick-rate and win-rate over recent N matches
- **Patch-aware features**: detect patch boundaries from data, encode patch era
- **Series context**: game position within series (game 1/2/3), prior hero selections in earlier games
- **Cross-team interaction**: head-to-head captain history, team matchup patterns

### 2. Model Architecture

- **LightGBM tuning**: Optuna hyperparameter search, feature importance → drop noise features
- **Neural models**: larger embeddings, attention over ban + team + captain tokens, pre-training on full draft prediction
- **Ensemble**: weighted average or stacking of LightGBM + Transformer predictions
- **Time-aware models**: sample weighting by recency, time-decay loss

### 3. Training Strategy

- **Temporal cross-validation**: sliding window splits respecting time order
- **Class balancing**: focal loss or oversampling for rare first-pick heroes
- **Data augmentation**: radiant ↔ dire symmetry (swap sides for each match → 2x data)
- **Multi-task learning**: jointly predict first pick + first ban + draft outcome

## Priority Order

1. Captain/team hero pool features + LightGBM tuning
2. Temporal features (rolling meta stats)
3. Data augmentation (side-swap)
4. Ensemble (LightGBM + Transformer)
5. Advanced training (multi-task, focal loss)
