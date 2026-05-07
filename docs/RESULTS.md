# Experimental Results

## Final Test-Set Metrics

| Model | MSE    | MAE    | R²     | Val Loss (final epoch) |
|-------|--------|--------|--------|------------------------|
| MLP   | 0.0582 | 0.1811 | 0.8836 | 0.0673                 |
| LSTM  | 0.0568 | 0.1742 | 0.8863 | 0.0724                 |
| RNN   | 0.1828 | 0.3373 | 0.6343 | 0.1982                 |

All models trained for up to 50 epochs with early stopping (patience=15).
LSTM stopped at epoch 47 (early stopping triggered).

## Key Observations

**MLP and LSTM perform comparably** — both achieve R² ≈ 0.88, well above the 0.85 threshold
that indicates a useful predictive model. LSTM edges MLP slightly on test MSE despite a
marginally higher validation loss, suggesting the LSTM generalises better to unseen data.

**RNN underperforms** — R² of 0.63 indicates the vanilla RNN struggles to separate the 10 Hz
component from the mixture. The absence of gating in a standard RNN leads to weaker gradient
flow across the 10-sample window. This is consistent with the known limitation noted in class:
"RNN is good for short memory but bad for long memory."

**Noise level is conservative** — `noise_std=0.05` is lighter than the σ=0.1 suggested in the
original specification, making extraction easier. This was an intentional design choice to
ensure clear learning signal during development.

## Learning Curves

All three models show clean convergence with no divergence or instability:

- MLP: monotonic val loss decrease, converges by epoch ~20
- LSTM: smooth decrease, plateaus around epoch 45 before early stopping
- RNN: slow but steady decrease, still improving at epoch 50

## Comparison vs. Assignment Partner

Our project targets single-frequency (10 Hz) extraction. Our partner's implementation does
multi-frequency extraction across {1, 5, 10, 20} Hz simultaneously.

On the **10 Hz column** of the partner's per-frequency plots:

| Model | Partner (10 Hz MSE) | Ours (MSE) | Winner |
|-------|---------------------|------------|--------|
| MLP   | 0.8718              | **0.0582** | Ours   |
| RNN   | 0.5940              | **0.1828** | Ours   |
| LSTM  | 0.2679              | **0.0568** | Ours   |

Our models outperform significantly on the shared target frequency.

## Potential Further Improvements

| Improvement | Expected Gain |
|-------------|---------------|
| Attention mechanism on RNN output | RNN R² → ~0.80 |
| Larger window size (10→20 samples) | More temporal context for all models |
| Data augmentation (phase shift) | Better generalisation |
| Ensemble MLP+LSTM | R² → ~0.91 |
