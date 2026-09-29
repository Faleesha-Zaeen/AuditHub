"""
AuditHub Page - Model Training
================================

Trains and compares candidate models, selects a winner and logs the run to
MLflow.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import ui
from app.components.session import has_repairs, require_dataset, set_target_column
from src.ml.trainer import MLTrainingEngine

ui.page_setup("Training", icon="◐")

ui.page_header(
    "Train and compare models",
    "The task type is inferred, three candidates are trained through the same "
    "preprocessing pipeline, and the best is kept and logged to MLflow.",
    eyebrow="Step 10 · Training",
)

df_to_train = require_dataset()
source_label = "repaired" if has_repairs() else "original"
st.caption(
    f"Training on the **{source_label}** dataset · {len(df_to_train):,} rows × "
    f"{len(df_to_train.columns)} columns"
)

missing_total = int(df_to_train.isna().sum().sum())
if missing_total:
    st.warning(
        f"{missing_total:,} missing cell(s) remain. The pipeline imputes them "
        "internally, but repairing first on **Repair** gives you control over how "
        "and keeps the exported dataset consistent with the model."
    )

control_column, result_column = st.columns([1, 2])

with control_column:
    target_col = st.selectbox("Target variable", df_to_train.columns.tolist())
    set_target_column(target_col)
    seed = st.number_input("Seed", value=42)

    distinct = df_to_train[target_col].nunique(dropna=True)
    st.caption(
        f"`{target_col}` has {distinct} distinct value(s) · "
        f"{'classification' if distinct <= 10 else 'likely regression'}"
    )

    if st.button("Train models", type="primary", width="stretch"):
        with st.spinner("Training candidates and selecting a winner…"):
            try:
                name = st.session_state.get("dataset_name", "dataset")
                model, metrics, save_path = MLTrainingEngine(seed=int(seed)).train(
                    df_to_train, target_column=target_col, dataset_name=name
                )
                st.session_state["training_results"] = {
                    "model": model, "metrics": metrics, "save_path": save_path,
                }
            except Exception as exc:
                st.error(f"Training failed: {exc}")

with result_column:
    results = st.session_state.get("training_results")
    if not results:
        ui.empty_state(
            "No model trained yet.",
            "Choose a target and press <strong>Train models</strong>.",
        )
    else:
        metrics = results["metrics"]
        scalar = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "feature_importances"
        }

        headline = [k for k in ("f1_score", "accuracy", "roc_auc", "rmse", "r2") if k in scalar]
        ui.stat_row([
            {"label": key.replace("_", " ").title(), "value": f"{scalar[key]:.4f}"}
            for key in headline[:4]
        ] or [{"label": "Metrics", "value": len(scalar)}])

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.caption(f"Best model saved to `models/{Path(results['save_path']).name}`")

        ui.dataframe(pd.DataFrame([
            {"Metric": k.replace("_", " ").title(), "Value": round(float(v), 6)}
            for k, v in scalar.items()
        ]))

        importances = metrics.get("feature_importances", {})
        if importances:
            ui.section("Feature importance", "From the winning model's own weights.")
            frame = (
                pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"])
                .sort_values("Importance", ascending=True)
                .tail(15)
            )
            figure = px.bar(
                frame, x="Importance", y="Feature", orientation="h",
                color_discrete_sequence=[ui.PLOT_COLORS["accent"]],
            )
            ui.plot(figure, height=max(260, 26 * len(frame)), show_legend=False)
            st.caption(
                "For per-prediction explanations and permutation importance, "
                "open **Explainability**."
            )

ui.divider()
st.caption(
    "Every run is logged to MLflow under the configured experiment, including "
    "parameters, metrics and plots."
)
