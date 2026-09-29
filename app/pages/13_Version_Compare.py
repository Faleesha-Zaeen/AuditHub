"""
AuditHub Page - Version Comparison & Lineage
=============================================

Compares two versions of a dataset and shows what the pipeline did to the data,
column by column.
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
from app.components.session import get_active_df, has_dataset
from src.ingestion.dataset_loader import DatasetLoader
from src.lineage.comparator import VersionComparator
from src.lineage.tracker import LineageTracker
from src.utils.constants import RAW_DATA_DIR
from src.utils.dataset_registry import DatasetRegistry

ui.page_setup("Versions", icon="◫")

ui.page_header(
    "What changed between versions?",
    "Row and column deltas, added, removed and renamed columns, type and "
    "missing-value changes, plus what the pipeline did to each column.",
    eyebrow="Versions & lineage",
)

_SEVERITY_COLOR = ui.STATUS_COLORS

compare_tab, lineage_tab, registry_tab = st.tabs(
    ["Compare versions", "Lineage", "Registry"]
)


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------

with compare_tab:
    ui.section("Compare two versions of a dataset")
    st.caption(
        "Pick the older version (V1) and the newer one (V2). Renames are only "
        "reported when the column contents actually match, never guessed from names."
    )

    accepted = sorted(ext.lstrip(".") for ext in DatasetLoader.supported_extensions())
    ingested = sorted(
        (p for p in RAW_DATA_DIR.glob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ) if RAW_DATA_DIR.exists() else []
    ingested_names = [p.name for p in ingested]

    left_col, right_col = st.columns(2)
    loader = DatasetLoader()

    def _pick(label: str, key: str):
        """Render a version picker returning (DataFrame, name)."""
        mode = st.radio(
            f"{label} source", ["Previously ingested", "Upload"],
            horizontal=True, key=f"{key}_mode",
        )
        if mode == "Upload":
            uploaded = st.file_uploader(label, type=accepted, key=f"{key}_upload")
            if uploaded is None:
                return None, ""
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            path = RAW_DATA_DIR / uploaded.name
            with open(path, "wb") as fh:
                fh.write(uploaded.getbuffer())
            try:
                return loader.load(path)[0], uploaded.name
            except Exception as exc:
                st.error(f"Could not load {uploaded.name}: {exc}")
                return None, ""

        if not ingested_names:
            st.info("No ingested files yet.")
            return None, ""
        choice = st.selectbox(label, ingested_names, key=f"{key}_select")
        try:
            return loader.load(RAW_DATA_DIR / choice)[0], choice
        except Exception as exc:
            st.error(f"Could not load {choice}: {exc}")
            return None, ""

    with left_col:
        ui.section("Version 1 (older)")
        left_df, left_name = _pick("Version 1", "v1")
    with right_col:
        ui.section("Version 2 (newer)")
        if has_dataset() and st.checkbox("Use the dataset loaded in this session", value=False):
            right_df = get_active_df()
            right_name = st.session_state.get("dataset_name", "current")
            st.success(f"Using **{right_name}** ({len(right_df):,} rows)")
        else:
            right_df, right_name = _pick("Version 2", "v2")

    opt1, opt2 = st.columns(2)
    with opt1:
        target_column = None
        if right_df is not None:
            options = ["None"] + [str(c) for c in right_df.columns]
            picked = st.selectbox("Target column (optional)", options)
            target_column = None if picked == "None" else picked
    with opt2:
        include_drift = st.checkbox(
            "Include distribution comparison", value=True,
            help="Runs the drift engine over shared columns. Turn off for a fast schema-only diff.",
        )

    if st.button("Compare versions", type="primary",
                 disabled=left_df is None or right_df is None):
        with st.spinner("Comparing versions..."):
            try:
                comparison = VersionComparator(detect_drift=include_drift).compare(
                    left_df, right_df,
                    left_name=left_name or "V1", right_name=right_name or "V2",
                    target_column=target_column,
                )
                st.session_state["version_comparison"] = comparison.to_dict()
                st.session_state["_version_result"] = comparison
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")

    comparison = st.session_state.get("_version_result")
    if comparison is not None:
        ui.divider()
        ui.section(f"{comparison.left_name} → {comparison.right_name}")

        def _shift(change: float, suffix: str = "", inverse: bool = False) -> str:
            """Render a V1 -> V2 change, coloured by whether it is an improvement."""
            if change == 0:
                return '<span style="color:var(--text-muted);">unchanged</span>'
            good = (change < 0) if inverse else (change > 0)
            colour = "var(--ok)" if good else "var(--warn)"
            return f'<span style="color:{colour};">{change:+,.2f}{suffix}</span>'.replace(
                ".00", ""
            )

        ui.stat_row([
            {"label": "Rows", "value": f"{comparison.right_rows:,}",
             "sub": _shift(comparison.row_delta)},
            {"label": "Columns", "value": comparison.right_columns,
             "sub": _shift(comparison.column_delta)},
            {"label": "Missing", "value": f"{comparison.right_missing_pct:.2f}%",
             "sub": _shift(comparison.right_missing_pct - comparison.left_missing_pct,
                           " pp", inverse=True)},
            {"label": "Duplicate rows", "value": f"{comparison.right_duplicates:,}",
             "sub": _shift(comparison.right_duplicates - comparison.left_duplicates,
                           inverse=True)},
        ])

        ui.dataframe(comparison.headline_frame())

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if comparison.added_columns:
                st.success("**Added**\n\n" + "\n".join(f"- `{c}`" for c in comparison.added_columns))
            else:
                st.caption("No columns added.")
        with sc2:
            if comparison.removed_columns:
                st.error("**Removed**\n\n" + "\n".join(f"- `{c}`" for c in comparison.removed_columns))
            else:
                st.caption("No columns removed.")
        with sc3:
            if comparison.renamed_columns:
                st.warning(
                    "**Renamed**\n\n" + "\n".join(
                        f"- `{r['from']}` → `{r['to']}` ({r['confidence']:.0%})"
                        for r in comparison.renamed_columns
                    )
                )
            else:
                st.caption("No renames detected.")

        ui.section("Column-level changes")
        for severity in ("CRITICAL", "WARNING", "INFO"):
            changes = comparison.changes_by_severity(severity)
            if not changes:
                continue
            for change in changes:
                text = f"**`{change.column}`** · {change.change_type} — {change.detail}"
                if severity == "CRITICAL":
                    st.error(text)
                elif severity == "WARNING":
                    st.warning(text)
                else:
                    st.info(text)

        if not comparison.column_changes:
            st.success("No column-level changes detected between these versions.")

        if comparison.drift:
            drifted = [c for c in comparison.drift["columns"] if c["status"] in ("WARNING", "CRITICAL")]
            if drifted:
                ui.section("Distribution shifts")
                chart = pd.DataFrame([
                    {"column": c["column"], "psi": c["psi"], "status": c["status"]}
                    for c in drifted
                ])
                ui.plot(
                    px.bar(chart, x="psi", y="column", orientation="h", color="status",
                           color_discrete_map=ui.STATUS_COLORS,
                           title="Columns whose distribution moved"),
                    height=max(260, 30 * len(chart)),
                )


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------

with lineage_tab:
    ui.section("What the pipeline did")
    st.caption(
        "Every headless pipeline run records what happened at each stage, "
        "including what was done to individual columns."
    )

    tracker = LineageTracker()
    try:
        runs = tracker.list_runs()
    except Exception as exc:
        runs = []
        st.error(f"Could not read lineage: {exc}")

    if not runs:
        st.info(
            "No runs recorded yet. Run the pipeline to populate lineage:\n\n"
            "`python -m src.pipeline run data/raw/your.csv --target your_label`"
        )
    else:
        labels = {
            f"{r['dataset_name'] or r['dataset_id'] or 'run'} · {r['events']} events · {r['run_id'][:8]}": r["run_id"]
            for r in runs
        }
        picked = st.selectbox("Pipeline run", list(labels.keys()))
        trace = tracker.trace(labels[picked])

        ui.section("Flow")
        st.markdown(
            " ".join(
                f"<span style='background:#eef2ff;padding:5px 12px;border-radius:14px;"
                f"margin-right:6px;display:inline-block;margin-bottom:6px;'>{s}</span>"
                for s in ["Raw Dataset"] + [x.replace("_", " ").title() for x in trace.stage_names]
            ),
            unsafe_allow_html=True,
        )

        ui.section("Stages")
        ui.dataframe(pd.DataFrame(trace.stages())[["stage", "event_count", "summary"]])

        columns = trace.columns()
        if columns:
            ui.section("Column history")
            chosen = st.selectbox("Column", sorted(columns.keys()))
            history = trace.column_history(chosen)

            st.markdown(f"**`{chosen}`**")
            for event in history:
                st.markdown(
                    f"<div style='margin-left:14px;border-left:2px solid #c7d2fe;"
                    f"padding:4px 0 4px 12px;'>→ {event.summary} "
                    f"<span style='color:#6b7280;font-size:0.8rem;'>[{event.stage}]</span></div>",
                    unsafe_allow_html=True,
                )

            ui.dataframe(trace.column_frame(chosen))

        if st.button("Attach this lineage to the report"):
            st.session_state["lineage"] = trace.to_dict()
            st.success("Lineage attached. Generate the report on the **Reports** page.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

with registry_tab:
    ui.section("Registered dataset versions")
    st.caption(
        "Uploads of the same logical dataset are grouped and numbered "
        "automatically, so repeated ingests become V1, V2, V3."
    )

    registry = DatasetRegistry()
    try:
        groups = registry.list_version_groups()
    except Exception as exc:
        groups = []
        st.error(f"Could not read the registry: {exc}")

    if not groups:
        st.info("Nothing registered yet. Upload a dataset on the **Upload** page.")
    else:
        ui.dataframe(pd.DataFrame(groups)[["version_group", "versions", "latest_version"]])

        group = st.selectbox("Show versions of", [g["version_group"] for g in groups])
        versions = registry.list_versions(group)
        if versions:
            ui.dataframe(
                pd.DataFrame([
                    {
                        "version": v.extra_metadata.get("version", 1),
                        "filename": v.filename,
                        "rows": v.num_rows,
                        "columns": v.num_columns,
                        "registered": v.upload_timestamp,
                        "dataset_id": v.dataset_id,
                    }
                    for v in versions
                ])
            )
