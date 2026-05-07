# PRD — MLP (Fully-Connected Network)
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Author: NajAmjad  | Date: 2026-05-07

---

## 1. Algorithm Overview

The Multi-Layer Perceptron (MLP) is a feedforward neural network with no temporal inductive bias. It treats the 10-sample noisy window as a flat feature vector and learns a direct nonlinear mapping from (noisy window + 1-hot frequency label) to the clean target window. While conceptually simple, the MLP serves as a strong baseline because it can in principle memorize the conditional amplitude/phase statistics of the 10 Hz component given sufficient capacity.

The absence of sequential structure is both a limitation and a strength: the MLP does not exploit the time-ordering of samples within a window, but it is not subject to vanishing-gradient problems across time steps.

---

## 2. Input / Output Contract

| Stage | Tensor Shape | Description |
|---|---|---|
| Input `x` | `(batch, window_size)` = `(B, 10)` | Noisy mixed signal window |
| Input `label` | `(batch, num_frequencies)` = `(B, 4)` | 1-hot frequency selector |
| Concatenated input | `(B, 14)` | `cat([x, label], dim=-1)` |
| Hidden layer outputs | `(B, H_i)` | Per-layer activations |
| Output | `(B, 10)` | Predicted clean 10 Hz window |

No reshaping or sequence axis is used.

---

## 3. Architecture Details

```
Input (14) → Linear(14, 1024) → LeakyReLU
           → Linear(1024, 512) → LeakyReLU
           → Linear(512, 256)  → LeakyReLU
           → Linear(256, 10)   [linear output, no activation]
```

**Parameter count formula:**

```
P = (14×1024 + 1024) + (1024×512 + 512) + (512×256 + 256) + (256×10 + 10)
  ≈ 14,336 + 524,800 + 131,328 + 2,570
  ≈ 673,034 parameters
```

Activation: `LeakyReLU` (configurable via `model.mlp.activation` in `config.yaml`). LeakyReLU avoids dying-ReLU problems in deep networks.

Hidden layer sizes are fully configurable: `model.mlp.hidden_sizes: [1024, 512, 256]`.

---

## 4. Training Considerations

- **No gradient clipping needed** by default (no recurrence, so no exploding gradients across time steps), though `grad_clip_max_norm` is still applied uniformly for consistency.
- **Regularization**: early stopping on validation MSE prevents overfitting.
- **Capacity**: the MLP is deliberately over-parameterized (~673K params) relative to the task complexity (10-sample windows) to ensure it is not capacity-limited.
- **Limitation**: treats each sample position independently; cannot explicitly model phase continuity across window positions.

---

## 5. Hyperparameters

| Parameter | Config Key | Default | Role |
|---|---|---|---|
| Hidden layer sizes | `model.mlp.hidden_sizes` | `[1024, 512, 256]` | Depth and width of the network |
| Activation function | `model.mlp.activation` | `leaky_relu` | Nonlinearity between layers |
| Learning rate | `training.learning_rate` | `0.01` | Adam optimizer step size |
| LR scheduler | `training.use_scheduler` | `true` | ReduceLROnPlateau on val loss |
| Gradient clipping | `training.grad_clip_max_norm` | `1.0` | Max gradient L2 norm (0 = disabled) |
| Early stopping | `training.early_stopping_patience` | `15` | Epochs without improvement before halt |

---

## 6. Expected Performance

The MLP typically achieves the **lowest R²** among the three architectures because it ignores temporal ordering. However, for short 10-sample windows at 200 Hz, the temporal context is limited — the MLP often performs comparably to the RNN, especially when the signal SNR is high.

Expected range (default config): **R² ≈ 0.80–0.90**, **MSE ≈ 0.02–0.08**.

---

## 7. Acceptance Criteria

- [ ] `MLPModel(config, window_size=10)` instantiates without error.
- [ ] `forward(x, label)` with `x.shape == (4, 10)`, `label.shape == (4, 4)` returns `(4, 10)`.
- [ ] No `nan` values in output after a forward pass on random inputs.
- [ ] `count_parameters()` returns a positive integer.
- [ ] `save(path)` writes a valid `.pt` checkpoint file.
- [ ] Test R² on held-out test set exceeds 0.70 after training.
