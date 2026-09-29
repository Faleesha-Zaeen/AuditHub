"""
AuditHub Page - Robustness
============================

Measures how far model performance decays as the data is progressively
corrupted, and summarises it as an elasticity score.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import ui
from app.components.session import data_source_caption, require_dataset, set_target_column

ui.page_setup("Robustness", icon="◎")

ui.page_header(
    "How much punishment can this data take?",
    "A model is retrained at escalating damage levels and its decay curve "
    "measured. The elasticity score summarises how well it holds up.",
    eyebrow="Step 9 · Robustness",
)

df = require_dataset()
data_source_caption()

control_column, result_column = st.columns([1, 2])

with control_column:
    target_col = st.selectbox("Target column", df.columns.tolist())
    set_target_column(target_col)

    mutation_type = st.selectbox(
        "Corruption type", ["missing_values", "gaussian_noise", "outliers"]
    )
    levels_str = st.text_input("Damage levels", "0.0, 0.05, 0.10, 0.25, 0.50")
    seed = st.number_input("Seed", min_value=1, value=42)

    st.caption(
        "Each level retrains a model, so a long list takes proportionally longer."
    )

    if st.button("Run robustness test", type="primary", width="stretch"):
        with st.spinner("Training baseline and perturbing…"):
            try:
                from src.robustness.evaluator import RobustnessEvaluator

                levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()]
                st.session_state["robustness_results"] = RobustnessEvaluator(
                    seed=int(seed)
                ).evaluate_robustness(
                    df, target_column=target_col,
                    mutation_type=mutation_type, levels=levels,
                )
            except Exception as exc:
                st.error(f"Robustness test failed: {exc}")

with result_column:
    results = st.session_state.get("robustness_results")
    if not results:
        ui.empty_state(
            "No robustness test run yet.",
            "Pick a target column and press <strong>Run robustness test</strong>.",
        )
    else:
        elasticity = float(results["elasticity_score"])
        if elasticity >= 0.85:
            verdict, status = "High", "STABLE"
        elif elasticity >= 0.60:
            verdict, status = "Medium", "WARNING"
        else:
            verdict, status = "Low", "CRITICAL"

        ui.stat_row([
            {"label": "Elasticity", "value": f"{elasticity:.4f}",
             "sub": ui.badge(f"{verdict} robustness", status)},
            {"label": "Baseline", "value": f"{float(results['performances'][0]):.4f}",
             "sub": str(results["metric_name"]).upper()},
            {"label": "Levels tested", "value": len(results["levels"])},
        ])

        frame = pd.DataFrame({
            "Damage (%)": [level * 100 for level in results["levels"]],
            "Performance ratio": results["normalized_performances"],
            "Raw score": results["performances"],
        })

        figure = px.line(
            frame, x="Damage (%)", y="Performance ratio", markers=True,
            color_discrete_sequence=[ui.PLOT_COLORS["accent"]],
            title=f"Decay curve · {str(results['metric_name']).upper()}",
        )
        figure.update_yaxes(range=[0.0, 1.1])
        figure.add_hline(
            y=1.0, line_dash="dash", line_color=ui.PLOT_COLORS["muted"],
            annotation_text="baseline",
        )
        ui.plot(figure, height=340, show_legend=False)

        ui.dataframe(frame.round(4))
