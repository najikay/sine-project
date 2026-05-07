# PRD — LSTM (Long Short-Term Memory)
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Author: najikay | Date: 2026-05-07

---

## 1. Algorithm Overview

The Long Short-Term Memory (LSTM) network extends the vanilla RNN with three multiplicative **gates** — input, forget, and output — that explicitly control what information is written to, erased from, and read out of a persistent **cell state**. This gating mechanism allows the LSTM to selectively retain relevant context across timesteps without suffering from vanishing gradients that plague vanilla RNNs.

For sine-wave extraction, the LSTM's cell state can learn to track phase continuity within the 10-sample window. By remembering where in the oscillation cycle the signal is, the LSTM can produce smoother, more phase-coherent reconstructions of the 10 Hz component than either the MLP or vanilla RNN.

Like the RNN, the LSTM operates in **bidirectional** mode by default, processing the window in both temporal directions simultaneously.

---

## 2. Input / Output Contract

| Stage | Tensor Shape | Description |
|---|---|---|
| Input `x` | `(B, 10)` | Noisy mixed signal window |
| Input `label` | `(B, 4)` | 1-hot frequency selector |
| Unsqueezed `x` | `(B, 10, 1)` | Add feature dimension |
| Expanded `label` | `(B, 10, 4)` | Repeat label per timestep |
| LSTM input | `(B, 10, 5)` | Concatenated signal + label |
| LSTM output | `(B, 10, 2×hidden)` | Bidirectional hidden states (cell state discarded) |
| Linear projection | `(B, 10, 1)` | Per-timestep projection |
| Squeezed output | `(B, 10)` | Predicted clean 10 Hz window |

---

## 3. Architecture Details

```
x: (B,10) → unsqueeze → (B,10,1)
label: (B,4) → expand → (B,10,4)
cat → (B,10,5)
  ↓
nn.LSTM(input_size=5, hidden_size=128, num_layers=2,
         batch_first=True, bidirectional=True)
  ↓ output: (B,10,256)  [h_n, c_n ignored]
nn.Linear(256, 1)
  ↓ squeeze → (B,10)
```

**Gate equations per timestep (single direction, single layer):**

```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # input gate
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # forget gate
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g) # cell candidate
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # output gate
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t      # cell state update
h_t = o_t ⊙ tanh(c_t)                 # hidden state output
```

**Parameter count formula (bidirectional, hidden=128, layers=2):**

```
Layer 1: 4 × (5×128 + 128×128 + 128) × 2 directions ≈ 264K
Layer 2: 4 × (256×128 + 128×128 + 128) × 2 directions ≈ 524K
Linear:  256×1 + 1 = 257
Total ≈ 788K parameters
```

LSTM has ~4× more parameters than vanilla RNN per layer (due to the 4 gates).

---

## 4. Training Considerations

- **Gradient clipping**: still applied (`grad_clip_max_norm: 1.0`) for robustness, though LSTM is inherently more stable than vanilla RNN due to the additive cell state update.
- **Cell state vs hidden state**: only the per-timestep hidden states (`output`) are used for prediction; the final `(h_n, c_n)` are discarded.
- **Bidirectionality**: same rationale as RNN — allows phase-aware reconstruction by attending to future context within the window.
- **Capacity advantage**: more parameters than vanilla RNN and more structured memory management, typically yielding the best reconstruction quality.

---

## 5. Hyperparameters

| Parameter | Config Key | Default | Role |
|---|---|---|---|
| Hidden size | `model.lstm.hidden_size` | `128` | Dimensionality of LSTM cell and hidden state |
| Number of layers | `model.lstm.num_layers` | `2` | Stacked LSTM depth |
| Bidirectional | `model.lstm.bidirectional` | `true` | Forward + backward pass through window |
| Learning rate | `training.learning_rate` | `0.01` | Adam optimizer step size |
| LR scheduler | `training.use_scheduler` | `true` | ReduceLROnPlateau on val loss |
| Gradient clipping | `training.grad_clip_max_norm` | `1.0` | Max gradient L2 norm |
| Early stopping | `training.early_stopping_patience` | `15` | Epochs without improvement before halt |

---

## 6. Expected Performance

The LSTM is the strongest of the three architectures for this task. Its gating mechanism enables more precise phase tracking across the 10-sample window, yielding lower MSE and higher R² than both MLP and vanilla RNN.

Expected range (default config): **R² ≈ 0.92–0.97**, **MSE ≈ 0.005–0.02**.

---

## 7. Acceptance Criteria

- [ ] `LSTMModel(config, window_size=10)` instantiates without error.
- [ ] `forward(x, label)` with `x.shape == (4, 10)`, `label.shape == (4, 4)` returns `(4, 10)`.
- [ ] No `nan` values in output after a forward pass on random inputs.
- [ ] `count_parameters()` returns a positive integer.
- [ ] Internal `self.lstm.hidden_size` matches `config.hidden_size`.
- [ ] LSTM output differs from RNN output on identical inputs (distinct architectures).
- [ ] Training with gradient clipping completes without `nan` losses even at `lr=0.1`.
- [ ] Test R² on held-out test set exceeds 0.85 after training.
