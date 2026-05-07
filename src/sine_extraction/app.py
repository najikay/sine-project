"""Interactive Streamlit dashboard for sine-wave extraction results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
import torch

matplotlib.use("Agg")

from sine_extraction.config_loader import load_config  # noqa: E402
from sine_extraction.constants import DEFAULT_CONFIG_PATH  # noqa: E402
from sine_extraction.models.lstm import LSTMModel  # noqa: E402
from sine_extraction.models.mlp import MLPModel  # noqa: E402
from sine_extraction.models.rnn import RNNModel  # noqa: E402
from sine_extraction.visualization.plotter import ComparisonPlotter  # noqa: E402

_ARTIFACTS = Path("artifacts")
_DATA_DIR = _ARTIFACTS / "data"
_CKPT_DIR = _ARTIFACTS / "checkpoints"
_RESULTS_DIR = _ARTIFACTS / "results"


def _load_models(cfg: object, device: torch.device) -> dict[str, object]:
    """Load trained checkpoints for all three architectures."""
    ws = cfg.signal.window_size  # type: ignore[attr-defined]
    models: dict[str, object] = {
        "mlp": MLPModel(cfg.model.mlp, ws),  # type: ignore[attr-defined]
        "rnn": RNNModel(cfg.model.rnn, ws),  # type: ignore[attr-defined]
        "lstm": LSTMModel(cfg.model.lstm, ws),  # type: ignore[attr-defined]
    }
    for _name, model in models.items():
        cls_name = model.__class__.__name__  # type: ignore[union-attr]
        pts = sorted(_CKPT_DIR.glob(f"{cls_name}*.pt"))
        if pts:
            model.load_state_dict(  # type: ignore[union-attr]
                torch.load(pts[-1], map_location=device)
            )
        model.eval()  # type: ignore[union-attr]
        model.to(device)  # type: ignore[union-attr]
    return models


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(page_title="Sine Extraction Dashboard", layout="wide")
    st.title("Sine Wave Extraction — Interactive Dashboard")

    # Sidebar controls
    st.sidebar.header("Settings")
    config_path = st.sidebar.text_input("Config path", str(DEFAULT_CONFIG_PATH))
    sample_idx = st.sidebar.number_input("Sample index", min_value=0, value=0, step=1)

    # Load config
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        st.error(f"Config not found: {cfg_path}")
        return
    cfg = load_config(cfg_path)

    # Check data exists
    if not (_DATA_DIR / "X.npy").exists():
        st.warning("No data found. Run `python -m sine_extraction generate` first.")
        return

    X = np.load(_DATA_DIR / "X.npy")
    y = np.load(_DATA_DIR / "y.npy")
    st.sidebar.info(f"Dataset: {len(X):,} windows loaded")

    sample_idx = min(int(sample_idx), len(X) - 1)  # type: ignore[arg-type]

    # Load models
    device = torch.device("cpu")
    if not _CKPT_DIR.exists() or not list(_CKPT_DIR.glob("*.pt")):
        st.warning("No checkpoints found. Run `python -m sine_extraction train` first.")
        return

    with st.spinner("Loading models…"):
        models = _load_models(cfg, device)

    # Build predictions
    freq_idx = cfg.signal.frequencies.index(cfg.signal.target_frequency)
    freq_label = torch.zeros(len(cfg.signal.frequencies))
    freq_label[freq_idx] = 1.0
    x_t = torch.tensor(X[sample_idx]).unsqueeze(0)
    label_t = freq_label.unsqueeze(0)

    with torch.no_grad():
        preds = {
            k: m(x_t, label_t).cpu().numpy()[0]  # type: ignore[operator]
            for k, m in models.items()
        }

    # Plot
    import tempfile
    vis_cfg = cfg.visualization
    with tempfile.TemporaryDirectory() as tmp:
        from dataclasses import replace
        tmp_vis = replace(vis_cfg, plots_dir=tmp, interactive=False)
        plotter = ComparisonPlotter(tmp_vis)
        plotter.plot(
            noisy=X[sample_idx],
            pure=y[sample_idx],
            mlp_pred=preds["mlp"],
            rnn_pred=preds["rnn"],
            lstm_pred=preds["lstm"],
            sample_idx=sample_idx,
        )
        st.pyplot(plotter.fig)

    # Metrics table
    metrics_path = _RESULTS_DIR / "metrics.json"
    if metrics_path.exists():
        st.subheader("Model Metrics (test set)")
        metrics = json.loads(metrics_path.read_text())
        st.dataframe(
            {
                "Model": list(metrics.keys()),
                "MSE": [v["mse"] for v in metrics.values()],
                "MAE": [v["mae"] for v in metrics.values()],
                "R²": [v["r2"] for v in metrics.values()],
            }
        )
    else:
        st.info("Run `python -m sine_extraction evaluate` to see metrics.")


if __name__ == "__main__":
    main()
