# PRD — RNN (Vanilla Recurrent Neural Network)
## Project: Sine Wave Extraction System
### Version: 1.0.0 | Author: najikay | Date: 2026-05-07

---

## 1. Algorithm Overview

The vanilla Recurrent Neural Network (RNN) processes the 10-sample noisy window as a time sequence. At each timestep, the RNN cell receives one signal sample (+ the 4-element 1-hot frequency label) and updates a hidden state that carries information forward from previous timesteps. This sequential inductive bias allows the RNN to model the temporal correlation inherent in a sine wave.

The key limitation of vanilla RNNs is susceptibility to **vanishing and exploding gradients** when backpropagating through long sequences. For 10-sample windows this risk is moderate, but gradient clipping (`grad_clip_max_norm: 1.0`) is applied to prevent instability during training.

A **bidirectional** configuration (default) processes the window in both forward and backward directions, doubling the hidden dimensionality at each timestep and allowing each output to attend to future context within the window.

---

## 2. Input / Output Contract

| Stage | Tensor Shape | Description |
|---|---|---|
| Input `x` | `(B, 10)` | Noisy mixed signal window |
| Input `label` | `(B, 4)` | 1-hot frequency selector |
| Unsqueezed `x` | `(B, 10, 1)` | Add feature dimension |
| Expanded `label` | `(B, 10, 4)` | Repeat label per timestep |
| RNN input | `(B, 10, 5)` | Concatenated signal + label |
| RNN output | `(B, 10, 2×hidden)` | Bidirectional hidden states |
| Linear projection | `(B, 10, 1)` | Per-timestep projection |
| Squeezed output | `(B, 10)` | Predicted clean 10 Hz window |

---

## 3. Architecture Details

```
x: (B,10) → unsqueeze → (B,10,1)
label: (B,4) → expand → (B,10,4)
cat → (B,10,5)
  ↓
nn.RNN(input_size=5, hidden_size=128, num_layers=2,
        batch_first=True, bidirectional=True)
  ↓ output: (B,10,256)
nn.Linear(256, 1)
  ↓ squeeze → (B,10)
```

**Parameter count formula (bidirectional, hidden=128, layers=2):**

```
Layer 1: 4 × (5×128 + 128×128 + 128) × 2 directions ≈ 132K
Layer 2: 4 × (256×128 + 128×128 + 128) × 2 directions ≈ 262K
Linear:  256×1 + 1 = 257
Total ≈ 394K parameters
```

Bidirectional and layer counts are configurable: `model.rnn.hidden_size`, `model.rnn.num_layers`, `model.rnn.bidirectional`.

---

## 4. Training Considerations

- **Gradient clipping is essential**: vanilla RNNs have unstable gradients. `clip_grad_norm_(params, max_norm=1.0)` is applied after every backward pass.
- **Vanishing gradients**: even with clipping, very long sequences would degrade performance — the 10-sample window is manageable for vanilla RNN.
- **Bidirectionality**: doubles capacity and allows output at timestep `t` to incorporate information from timesteps `t+1 ... T-1`, improving phase-coherent reconstruction.
- **Early stopping**: halts when validation MSE stops improving for `early_stopping_patience` epochs.

---

## 5. Hyperparameters

| Parameter | Config Key | Default | Role |
|---|---|---|---|
| Hidden size | `model.rnn.hidden_size` | `128` | Dimensionality of RNN hidden state |
| Number of layers | `model.rnn.num_layers` | `2` | Stacked RNN depth |
| Bidirectional | `model.rnn.bidirectional` | `true` | Forward + backward pass through window |
| Learning rate | `training.learning_rate` | `0.01` | Adam optimizer step size |
| LR scheduler | `training.use_scheduler` | `true` | ReduceLROnPlateau on val loss |
| Gradient clipping | `training.grad_clip_max_norm` | `1.0` | Max gradient L2 norm |
| Early stopping | `training.early_stopping_patience` | `15` | Epochs without improvement before halt |

---

## 6. Expected Performance

The RNN exploits temporal ordering of the 10-sample window and typically outperforms the MLP baseline. However, vanilla RNN hidden state updates involve a single tanh nonlinearity, which provides limited memory gating compared to LSTM. As a result, the RNN usually achieves **intermediate performance** between MLP and LSTM.

Expected range (default config): **R² ≈ 0.88–0.95**, **MSE ≈ 0.01–0.04**.

---

## 7. Acceptance Criteria

- [ ] `RNNModel(config, window_size=10)` instantiates without error.
- [ ] `forward(x, label)` with `x.shape == (4, 10)`, `label.shape == (4, 4)` returns `(4, 10)`.
- [ ] No `nan` values in output after a forward pass on random inputs.
- [ ] `count_parameters()` returns a positive integer.
- [ ] Internal `self.rnn.hidden_size` matches `config.hidden_size`.
- [ ] Training with gradient clipping completes without `nan` losses even at `lr=0.1`.
- [ ] Test R² on held-out test set exceeds 0.80 after training.
