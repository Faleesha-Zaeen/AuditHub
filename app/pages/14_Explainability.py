"""
AuditHub Page - Model Explainability
======================================

What the model learned overall, and why it made one specific prediction.
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
from app.components.session import has_repairs, require_dataset, set_target_column
from src.ml.explainer import ModelExplainer
from src.ml.trainer import MLTrainingEngine

ui.page_setup("Explainability", icon="◍")

ui.page_header(
    "Why did the model decide that?",
    "Global importance measured by permuting the full pipeline, and per-prediction "
    "contributions from SHAP — both reported against your original columns, not "
    "one-hot fragments.",
    eyebrow="Step 11 · Explainability",
)

df = require_dataset()
st.caption(
    f"Using the **{'repaired' if has_repairs() else 'original'}** dataset · "
    f"{len(df):,} rows × {len(df.columns)} columns"
)


def _display_value(value) -> str:
    """Render a single cell as text for display.

    A row spans every column type at once, and Streamlit serialises tables
    through Arrow, which rejects an object column mixing ints, strings and
    timestamps.
    """
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

trained = st.session_state.get("training_results")

setup_left, setup_right = st.columns([2, 1])
with setup_left:
    target_col = st.selectbox("Target column", df.columns.tolist())
    set_target_column(target_col)
    repeats = st.slider(
        "Permutation repeats", 3, 20, 5,
        help="More repeats give a steadier importance ranking but take longer.",
    )
with setup_right:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    if trained:
        st.caption("A trained model is available from this session.")
        reuse = st.checkbox("Reuse trained model", value=True)
    else:
        reuse = False
        st.caption("No model in this session — one will be trained now.")

    run = st.button("Explain model", type="primary", width="stretch")

if run:
    with st.spinner("Preparing model and computing explanations…"):
        try:
            if reuse and trained:
                model = trained["model"]
                metrics = trained["metrics"]
            else:
                name = st.session_state.get("dataset_name", "dataset")
                model, metrics, save_path = MLTrainingEngine().train(
                    df, target_column=target_col, dataset_name=name
                )
                st.session_state["training_results"] = {
                    "model": model, "metrics": metrics, "save_path": save_path,
                }

            task_type = "classification" if "f1_score" in metrics else "regression"
            frame = df.dropna(subset=[target_col])
            X = frame.drop(columns=[target_col])
            y = frame[target_col]

            explainer = ModelExplainer(model, task_type=task_type)
            report = explainer.explain(
                X, y, rows=[0], target_column=target_col, n_repeats=repeats
            )

            st.session_state["_explainer"] = explainer
            st.session_state["_explain_X"] = X
            st.session_state["_explain_task"] = task_type
            st.session_state["explainability"] = report.to_dict()
            st.session_state["_explain_report"] = report
        except Exception as exc:
            st.error(f"Explanation failed: {exc}")

report = st.session_state.get("_explain_report")
if report is None:
    ui.divider()
    ui.empty_state(
        "No explanation yet.",
        "Choose a target column and press <strong>Explain model</strong>.",
    )
    st.stop()

explainer = st.session_state["_explainer"]
X = st.session_state["_explain_X"]
task_type = st.session_state["_explain_task"]

ui.divider()

global_tab, local_tab = st.tabs(["Global importance", "Explain a prediction"])

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------

with global_tab:
    ui.section(
        "What the model relies on",
        "Measured by permuting each column through the whole fitted pipeline, so "
        "the score belongs to your column rather than an encoded fragment of it.",
    )

    top = report.top_features(20)
    if not top:
        ui.empty_state("No global importance could be computed for this model.")
    else:
        leader = top[0]
        ui.stat_row([
            {"label": "Most important", "value": leader.feature,
             "sub": f"{leader.importance:+.4f}"},
            {"label": "Method",
             "value": report.method.split(" + ")[0].replace("_", " ").title(),
             "sub": "shap" if "shap" in report.method else "occlusion"},
            {"label": "Features ranked", "value": len(report.global_importance)},
            {"label": "Task", "value": task_type.title()},
        ])

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        frame = pd.DataFrame([
            {"feature": f.feature, "importance": f.importance, "std": f.std}
            for f in reversed(top)
        ])
        figure = px.bar(
            frame, x="importance", y="feature", orientation="h",
            error_x="std",
            color_discrete_sequence=[ui.PLOT_COLORS["accent"]],
            labels={"importance": "Permutation importance", "feature": ""},
        )
        ui.plot(figure, height=max(280, 28 * len(frame)), show_legend=False)

        ui.dataframe(report.importance_frame().round(6))

# ---------------------------------------------------------------------------
# Local
# ---------------------------------------------------------------------------

with local_tab:
    ui.section(
        "Explain one prediction",
        "Pick a row to see which features pushed the model toward its answer, "
        "and which argued against it.",
    )

    picker_left, picker_right = st.columns([1, 2])
    with picker_left:
        row_index = st.number_input(
            "Row", min_value=0, max_value=max(len(X) - 1, 0), value=0, step=1
        )
        top_n = st.slider("Features to show", 3, 20, 8)
        explain_row = st.button("Explain this row", type="primary", width="stretch")
    with picker_right:
        st.caption("Row values")
        chosen_row = X.iloc[int(row_index)]
        ui.dataframe(
            pd.DataFrame({
                "Feature": [str(c) for c in X.columns],
                # Rendered as text: one row spans every column type, and Arrow
                # cannot serialise a mixed int/str/Timestamp object column.
                "Value": [_display_value(chosen_row[c]) for c in X.columns],
                "Type": [str(X[c].dtype) for c in X.columns],
            })
        )

    if explain_row:
        with st.spinner("Computing contributions…"):
            try:
                st.session_state["_local_explanation"] = explainer.explain_row(
                    X, int(row_index), background=X, top_n=top_n
                )
            except Exception as exc:
                st.error(f"Could not explain this row: {exc}")

    local = st.session_state.get("_local_explanation")
    if local is None:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        ui.empty_state("Pick a row and press Explain this row.")
    else:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        confidence = (
            f"{local.confidence:.1%} confidence" if local.confidence is not None
            else "regression output"
        )
        st.markdown(
            f"""
            <div class="ah-card">
                <div class="ah-card-title">Prediction · row {local.row_index}</div>
                <div style="font-family:var(--font-mono);font-size:1.8rem;
                            font-weight:600;color:var(--text);">
                    {local.prediction}
                </div>
                <div style="font-size:0.84rem;color:var(--text-muted);margin-top:4px;">
                    {confidence}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        toward = local.pushing_toward()
        against = local.pushing_against()

        summary_left, summary_right = st.columns(2)
        with summary_left:
            body = "".join(
                f'<div style="font-size:0.86rem;color:var(--text-secondary);'
                f'padding:5px 0;border-bottom:1px solid var(--border);">'
                f'<code>{c.feature}</code> = {_display_value(c.value)} '
                f'<span style="float:right;font-family:var(--font-mono);'
                f'color:var(--danger);">{c.contribution:+.4f}</span></div>'
                for c in toward[:5]
            ) or '<div style="color:var(--text-muted);font-size:0.85rem;">none</div>'
            ui.card("Pushed toward this outcome", body)
        with summary_right:
            body = "".join(
                f'<div style="font-size:0.86rem;color:var(--text-secondary);'
                f'padding:5px 0;border-bottom:1px solid var(--border);">'
                f'<code>{c.feature}</code> = {_display_value(c.value)} '
                f'<span style="float:right;font-family:var(--font-mono);'
                f'color:var(--ok);">{c.contribution:+.4f}</span></div>'
                for c in against[:5]
            ) or '<div style="color:var(--text-muted);font-size:0.85rem;">none</div>'
            ui.card("Argued against it", body)

        ui.section("Contribution breakdown")
        ui.contribution_bars([
            {**c.to_dict(), "value": _display_value(c.value)}
            for c in local.contributions
        ])

        with st.expander("Plain-language summary", expanded=False):
            st.code(local.narrative(), language=None)

ui.divider()
st.caption(
    "The explanation is attached to the session and included automatically in the "
    "consolidated report generated on **Reports**."
)
