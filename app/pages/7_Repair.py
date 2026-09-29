"""
AuditHub Page - Repair
========================

Whole-dataset auto-repair, per-column manual repair with previews, an undo
stack, and a verified export.

Repairs made here become the active dataset for every other page, so the
Quality Audit, Health Score and Reports all recompute against repaired data.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import ui
from app.components.session import (
    get_original_df,
    get_repairer,
    has_repairs,
    notify_data_changed,
    require_dataset,
    reset_repairer,
    target_selector,
)

ui.page_setup("Repair", icon="◇")

ui.page_header(
    "Repair the dataset",
    "Fill gaps, drop duplicates and fix types — then every other page recomputes "
    "against the repaired data. Labels are never imputed.",
    eyebrow="Step 7 · Repair",
)

require_dataset()
repairer = get_repairer()
original_df = get_original_df()
current = repairer.df


def _missing_overview(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column missing-value table."""
    total = len(frame)
    rows = [
        {
            "Column": str(col),
            "Type": str(frame[col].dtype),
            "Missing": int(frame[col].isna().sum()),
            "Missing %": round(int(frame[col].isna().sum()) / total * 100, 2) if total else 0.0,
        }
        for col in frame.columns
        if int(frame[col].isna().sum()) > 0
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

missing_now = int(current.isna().sum().sum())
missing_before = int(original_df.isna().sum().sum())


def _delta(now: int, before: int, inverse: bool = False) -> str:
    """Render a signed change, coloured by whether the move is an improvement."""
    change = now - before
    if change == 0:
        return '<span style="color:var(--text-muted);">unchanged</span>'
    good = (change < 0) if inverse else (change > 0)
    colour = "var(--ok)" if good else "var(--warn)"
    return f'<span style="color:{colour};">{change:+,} vs original</span>'


ui.stat_row([
    {"label": "Rows", "value": f"{len(current):,}",
     "sub": _delta(len(current), len(original_df))},
    {"label": "Columns", "value": len(current.columns),
     "sub": _delta(len(current.columns), len(original_df.columns))},
    {"label": "Missing cells", "value": f"{missing_now:,}",
     "sub": _delta(missing_now, missing_before, inverse=True)},
    {"label": "Repair steps", "value": len(repairer.get_log()),
     "sub": ui.badge("applied", "STABLE") if repairer.get_log() else "none yet"},
])

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
if has_repairs():
    st.success("Repairs applied. Every other page is now analysing the repaired dataset.")
else:
    st.info("No repairs yet. Other pages are analysing the original upload.")

ui.divider()

# ---------------------------------------------------------------------------
# Auto-repair
# ---------------------------------------------------------------------------

ui.section(
    "Auto-repair",
    "One pass over the whole dataset: drop empty columns and duplicate rows, "
    "then impute every remaining gap by column type.",
)

settings_column, action_column = st.columns([2, 1])

with settings_column:
    target_column = target_selector(
        current,
        label="Target column (optional)",
        help_text=(
            "The label you intend to predict. It is never imputed — rows with a "
            "missing label are dropped instead, because inventing labels fabricates "
            "ground truth."
        ),
    )
    option_left, option_right = st.columns(2)
    with option_left:
        numeric_strategy = st.selectbox(
            "Numeric columns", ["median", "mean", "mode"], index=0,
            help="Median is the default: unlike the mean it is not dragged by outliers.",
        )
    with option_right:
        categorical_strategy = st.selectbox("Text columns", ["mode", "constant"], index=0)

    toggle_left, toggle_right = st.columns(2)
    with toggle_left:
        drop_dupes = st.checkbox("Remove duplicate rows", value=True)
    with toggle_right:
        drop_empty = st.checkbox("Drop fully empty columns", value=True)

with action_column:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    if st.button("Auto-Repair everything", type="primary", width="stretch"):
        with st.spinner("Repairing…"):
            try:
                result = repairer.auto_repair(
                    target_column=target_column,
                    numeric_strategy=numeric_strategy,
                    categorical_strategy=categorical_strategy,
                    remove_duplicate_rows=drop_dupes,
                    drop_empty_columns=drop_empty,
                )
                notify_data_changed()
                st.session_state["auto_repair_result"] = result.to_dict()
                st.rerun()
            except Exception as exc:
                st.error(f"Auto-repair failed: {exc}")

    if st.button("Reset to original", width="stretch"):
        reset_repairer()
        st.rerun()

result = st.session_state.get("auto_repair_result")
if result:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ui.stat_row([
        {"label": "Columns imputed", "value": len(result["columns_imputed"])},
        {"label": "Duplicates removed", "value": result["duplicates_removed"]},
        {"label": "Rows dropped", "value": result["target_rows_dropped"],
         "sub": "missing label"},
        {"label": "Columns skipped", "value": len(result["skipped"])},
    ])

    if result["columns_imputed"]:
        with st.expander("What each column was filled with", expanded=True):
            ui.dataframe(pd.DataFrame([
                {
                    "Column": column,
                    "Strategy": info["strategy"],
                    # Rendered as text: fill values span numbers, strings and
                    # dates, and Arrow refuses a mixed-type object column.
                    "Filled with": "—" if info["fill_value"] is None
                                   else str(info["fill_value"]),
                    "Cells": int(info["cells_filled"]),
                    "Note": info.get("note") or "",
                }
                for column, info in result["columns_imputed"].items()
            ]))
    if result["skipped"]:
        st.warning("Skipped:\n\n" + "\n".join(f"- {s}" for s in result["skipped"]))

# ---------------------------------------------------------------------------
# Remaining gaps
# ---------------------------------------------------------------------------

ui.divider()
ui.section("Missing values")

missing_table = _missing_overview(repairer.df)
if missing_table.empty:
    st.success("No missing values remain.")
else:
    chart_column, table_column = st.columns([1, 1])
    with chart_column:
        figure = px.bar(
            missing_table.sort_values("Missing %"),
            x="Missing %", y="Column", orientation="h",
            color_discrete_sequence=[ui.PLOT_COLORS["warn"]],
            labels={"Column": ""},
        )
        ui.plot(figure, height=max(240, 30 * len(missing_table)), show_legend=False)
    with table_column:
        ui.dataframe(missing_table, height=max(240, 30 * len(missing_table)))

# ---------------------------------------------------------------------------
# Manual repair
# ---------------------------------------------------------------------------

ui.divider()
ui.section("Manual repair", "Column-level control, with a preview before anything is applied.")

control_column, preview_column = st.columns([1, 2])

with control_column:
    action_type = st.selectbox(
        "Operation",
        [
            "Impute missing values",
            "Remove duplicates",
            "Coerce column type",
            "Scale numeric column",
            "Drop constant columns",
            "Clamp outliers",
            "Group rare categories",
            "Encode categorical",
        ],
    )

    apply_disabled = False
    repair_args: dict = {}
    repair_method = ""

    frame = repairer.df
    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c].dtype)]
    other_cols = [c for c in frame.columns if c not in numeric_cols]

    if action_type == "Remove duplicates":
        st.caption(f"{int(frame.duplicated().sum())} duplicate row(s) present.")
        repair_method = "remove_duplicates"

    elif action_type == "Impute missing values":
        with_nulls = [c for c in frame.columns if frame[c].isna().any()]
        if not with_nulls:
            st.caption("Nothing to impute — no column has missing values.")
            apply_disabled = True
        else:
            column = st.selectbox("Column", with_nulls)
            st.caption(
                f"{int(frame[column].isna().sum())} missing of {len(frame)} · {frame[column].dtype}"
            )
            strategy = st.selectbox("Strategy", ["median", "mean", "mode", "constant"])
            fill_value = None
            if strategy == "constant":
                fill_value = st.text_input("Constant value")
                if not str(fill_value).strip():
                    st.caption("Enter a value to enable Apply.")
                    apply_disabled = True
            repair_method = "impute_missing"
            repair_args = {"column": column, "strategy": strategy, "fill_value": fill_value}

    elif action_type == "Coerce column type":
        column = st.selectbox("Column", list(frame.columns))
        target_type = st.selectbox("Target type", ["int", "float", "datetime", "str"])
        repair_method = "coerce_types"
        repair_args = {"column": column, "target_type": target_type}

    elif action_type == "Scale numeric column":
        if not numeric_cols:
            st.caption("No numeric columns available.")
            apply_disabled = True
        else:
            column = st.selectbox("Column", numeric_cols)
            method = st.selectbox("Method", ["standard", "minmax"])
            repair_method = "scale_numeric"
            repair_args = {"column": column, "method": method}

    elif action_type == "Drop constant columns":
        constant = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
        st.caption(f"{len(constant)} constant column(s) detected.")
        repair_method = "drop_constant_columns"

    elif action_type == "Clamp outliers":
        if not numeric_cols:
            st.caption("No numeric columns available.")
            apply_disabled = True
        else:
            column = st.selectbox("Column", numeric_cols)
            method = st.selectbox("Method", ["iqr", "zscore"])
            repair_method = "clamp_outliers"
            repair_args = {"column": column, "method": method}

    elif action_type == "Group rare categories":
        if not other_cols:
            st.caption("No categorical columns available.")
            apply_disabled = True
        else:
            column = st.selectbox("Column", other_cols)
            threshold = st.slider("Frequency threshold", 0.005, 0.10, 0.01, 0.005)
            repair_method = "handle_rare_categories"
            repair_args = {"column": column, "threshold": threshold}

    elif action_type == "Encode categorical":
        if not other_cols:
            st.caption("No categorical columns available.")
            apply_disabled = True
        else:
            column = st.selectbox("Column", other_cols)
            repair_method = "encode_categorical"
            repair_args = {"column": column, "method": "ordinal"}

    button_left, button_right = st.columns(2)
    with button_left:
        preview_clicked = st.button("Preview", disabled=apply_disabled, width="stretch")
    with button_right:
        apply_clicked = st.button(
            "Apply", type="primary", disabled=apply_disabled, width="stretch"
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Undo last step", disabled=not repairer.get_log(), width="stretch"):
        repairer.revert()
        notify_data_changed()
        st.rerun()

with preview_column:
    if apply_clicked and repair_method:
        with st.spinner("Applying…"):
            try:
                getattr(repairer, repair_method)(**repair_args)
                notify_data_changed()
                st.rerun()
            except Exception as exc:
                st.error(f"Repair failed: {exc}")

    if preview_clicked and repair_method:
        try:
            preview = repairer.preview_repair(repair_method, **repair_args)
            before_nulls = int(repairer.df.isna().sum().sum())
            after_nulls = int(preview.isna().sum().sum())

            st.markdown(
                f'<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:8px;">'
                f"Missing cells {before_nulls:,} &rarr; {after_nulls:,} "
                f"({after_nulls - before_nulls:+,}) · "
                f"rows {len(repairer.df):,} &rarr; {len(preview):,}</div>",
                unsafe_allow_html=True,
            )
            left, right = st.columns(2)
            with left:
                st.caption("Current")
                ui.dataframe(repairer.df.head(14))
            with right:
                st.caption("After this repair")
                ui.dataframe(preview.head(14))
        except Exception as exc:
            st.error(f"Could not build a preview: {exc}")
    else:
        st.caption("Current dataset")
        ui.dataframe(repairer.df.head(16))

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

ui.divider()
ui.section("Export")

dataset_name = st.session_state.get("dataset_name", "dataset")
target_column = st.session_state.get("target_column")
issues = repairer.verify_clean(target_column=target_column)

if issues:
    st.warning(
        "The dataset does not yet meet the export guarantees:\n\n"
        + "\n".join(f"- {issue}" for issue in issues)
        + "\n\nRun **Auto-Repair** above to resolve them."
    )
else:
    st.success(
        "All export guarantees met: no missing values, no duplicate rows, no residual "
        "missing-value markers, correct column types, and no index column."
    )

export_left, export_middle, export_right = st.columns(3)
stem = dataset_name.rsplit(".", 1)[0]

with export_left:
    st.download_button(
        "Download repaired CSV",
        data=repairer.to_csv_bytes(),
        file_name=f"{stem}_repaired.csv",
        mime="text/csv",
        width="stretch",
    )
with export_middle:
    manifest = repairer.build_manifest(dataset_name=dataset_name, target_column=target_column)
    st.download_button(
        "Download manifest",
        data=json.dumps(manifest, indent=2, default=str),
        file_name=f"{stem}_manifest.json",
        mime="application/json",
        width="stretch",
    )
with export_right:
    if st.button("Save to data/repaired", width="stretch"):
        try:
            csv_path, _ = repairer.export(
                dataset_name=dataset_name, target_column=target_column
            )
            st.success(f"Saved `{csv_path.name}`.")
        except Exception as exc:
            st.error(f"Export failed: {exc}")

# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

logs = repairer.get_log()
if logs:
    ui.divider()
    ui.section("Repair log", "Every applied step, in order.")
    ui.dataframe(pd.DataFrame([
        {
            "Step": entry["step"],
            "Action": entry["action"],
            "Details": ", ".join(f"{k}={v}" for k, v in entry["details"].items()),
            "At": entry["timestamp"][11:19],
        }
        for entry in logs
    ]))
