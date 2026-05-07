# PROMPT_LOG — AI Prompt Engineering Log
## HW1: Sine Wave Extraction with MLP, RNN, and LSTM

This document logs the key prompts used during development with Claude AI (claude-sonnet-4-6),
including the context, output, and lessons learned from each interaction.

---

## Prompt 1 — Project Scaffolding
**Context:** Starting HW1, needed a full project skeleton before writing any model code.  
**Prompt:** "Create a Python project for sine wave extraction from a noisy multi-frequency
signal. Use MLP, RNN, and LSTM. Frequencies: 1, 5, 10, 20 Hz. Window size 10 samples.
MSE loss. Must have ruff 0 violations, ≥85% test coverage, ≤150 lines per file."  
**Output:** Full src/sine_extraction/ package with data/, models/, training/, evaluation/,
visualization/, and cli/ subpackages. pyproject.toml, config/config.yaml, tests/.  
**Lesson:** Specifying ALL quality constraints upfront (line limits, coverage, ruff) in the
first prompt avoids expensive rewrites later.

---

## Prompt 2 — TDD Test Suite
**Context:** Needed tests written before implementation to follow TDD strictly.  
**Prompt:** "Write tests first for all modules: data generator, dataset, splitter, MLP, RNN,
LSTM, trainer with early stopping, metrics (MSE, MAE, R²), and CLI commands. Use pytest
fixtures. Cover edge cases."  
**Output:** 20 test files covering all modules. Fixtures in conftest.py shared across tests.  
**Lesson:** TDD revealed API mismatches early — the SineDataset initially expected 2 args
but the forward pass needed 3. Caught before implementation.

---

## Prompt 3 — Frequency Label Conditioning
**Context:** Needed models to extract a specific frequency from the mixture, not just denoise.  
**Prompt:** "Each model should accept a one-hot frequency label alongside the input window.
For RNN and LSTM, broadcast the label to every timestep. For MLP, concatenate with the
flattened input."  
**Output:** All three models updated with label conditioning. RNN/LSTM broadcast label via
`label.unsqueeze(1).expand(-1, window_size, -1)` at every step.  
**Lesson:** Broadcasting the label to every RNN/LSTM timestep consistently outperforms
concatenating only at the final hidden state — the model has frequency context at each step.

---

## Prompt 4 — Gradient Clipping & Early Stopping
**Context:** RNN training was unstable — exploding gradients causing NaN losses.  
**Prompt:** "Add gradient clipping (max_norm=1.0) to the trainer after backward(). Also add
early stopping with configurable patience."  
**Output:** trainer.py updated with `torch.nn.utils.clip_grad_norm_` and patience counter.  
**Lesson:** Gradient clipping is essential for RNNs. Without it, training diverges within
the first 5 epochs on this task. max_norm=1.0 stabilises all three models.

---

## Prompt 5 — Line Count Compliance
**Context:** trainer.py grew to 158 lines, violating the ≤150 line rule.  
**Prompt:** "trainer.py is 158 lines. Extract checkpoint saving logic into a separate
training/checkpoint_manager.py module. Update trainer.py to import from it."  
**Output:** checkpoint_manager.py created (28 lines). trainer.py reduced to 141 lines.  
**Lesson:** Design for the line limit from the start. Functions that write to disk
(checkpoints, metrics, plots) are natural extraction candidates.

---

## Prompt 6 — Hyperparameter Tuning from Partner's Project
**Context:** Partner's project achieved 6x lower MSE with a smaller MLP architecture
[64, 128, 64] and LSTM dropout=0.2. Merged these improvements.  
**Prompt:** "Update config.yaml: MLP hidden_sizes [64,128,64], RNN nonlinearity relu,
LSTM dropout 0.2. Update types.py and models to support these new config fields."  
**Output:** Config updated, LSTMConfig and RNNConfig extended, models updated.  
**Lesson:** Smaller models generalise better when training data is limited.
Dropout is essential for multi-layer LSTMs to prevent overfitting.

---

## Prompt 7 — Streamlit Dashboard
**Context:** Lecture notes mentioned "a visual UI for the signals would be nice."  
**Prompt:** "Add a Streamlit dashboard that lets the user pick a sample index and
visualise the noisy input, ground truth, and all three model predictions side by side."  
**Output:** src/sine_extraction/app.py with interactive slider and 6-panel plot.  
**Lesson:** Adding a UI beyond the minimum requirements demonstrates understanding of the
problem and scores extra on documentation/presentation quality.

---

## Best Practices Identified

1. **State all quality constraints in the first prompt** — line limits, coverage, linting rules.
2. **Write tests before implementation** — TDD reveals interface mismatches early.
3. **Broadcast labels at every timestep** — not just the final hidden state.
4. **Always use gradient clipping for RNNs** — `max_norm=1.0` prevents instability.
5. **Prefer smaller models with regularisation** — dropout + smaller hidden sizes generalise better.
6. **Extract IO-heavy functions** — checkpoints, plots, metrics naturally fit separate modules.
7. **Document prompt iterations** — shows the reasoning process behind architectural decisions.
