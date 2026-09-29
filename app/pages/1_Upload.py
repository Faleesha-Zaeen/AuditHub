"""
AuditHub Page - Dataset Ingestion
===================================

Uploads, parses, cleans and registers a dataset for the rest of the session.
"""

# Ensure the project root is importable when Streamlit runs this file as a
# standalone script (it puts the page's own directory on sys.path, not the root).
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from app.components import ui
from app.components.session import (
    get_original_df,
    get_repairer,
    has_dataset,
    has_repairs,
    notify_data_changed,
    set_dataset,
)
from src.ingestion.dataset_analyzer import DatasetAnalyzer
from src.ingestion.dataset_loader import DatasetLoader
from src.utils.constants import RAW_DATA_DIR
from src.utils.dataset_metadata import DatasetMetadata
from src.utils.dataset_registry import DatasetRegistry

ui.page_setup("Upload", icon="↑")

ui.page_header(
    "Ingest a dataset",
    "CSV, TSV, JSON, Excel or Parquet. Delimiters, encodings and Excel sheets "
    "are detected on load, and disguised missing values become real nulls before "
    "anything is measured.",
    eyebrow="Step 1 · Ingestion",
)

# Keep the picker in step with what the loader can actually parse, so it never
# accepts a file the loader will then reject.
_ACCEPTED = sorted(ext.lstrip(".") for ext in DatasetLoader.supported_extensions())

uploaded_file = st.file_uploader("Dataset file", type=_ACCEPTED)

if uploaded_file is not None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())

    loader = DatasetLoader()

    # Excel workbooks may hold several sheets; let the user pick rather than
    # silently reading the first one.
    sheet_name = None
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        sheets = loader.list_sheets(file_path)
        if len(sheets) > 1:
            sheet_name = st.selectbox(
                "This workbook has multiple sheets. Which holds your data?", sheets
            )

    option_left, option_right = st.columns(2)
    with option_left:
        apply_cleaning = st.checkbox(
            "Clean on load",
            value=True,
            help=(
                "Convert disguised missing values ('?', 'N/A', '-', blanks) into real "
                "nulls, trim column names and text, turn numeric-looking text into "
                "numbers, and drop fully empty rows and columns."
            ),
        )
    with option_right:
        row_limit = st.number_input(
            "Row limit (0 = no limit)", min_value=0, value=0, step=1000,
            help="Read only the first N rows. Useful for very large files.",
        )

    with st.spinner("Parsing and analysing…"):
        try:
            df, file_meta = loader.load(
                file_path,
                clean=apply_cleaning,
                sheet_name=sheet_name,
                max_rows=int(row_limit) or None,
            )

            summary = DatasetAnalyzer().analyze(df, source_filename=uploaded_file.name)

            registry = DatasetRegistry()
            meta = DatasetMetadata(
                filename=uploaded_file.name,
                shape=(len(df), len(df.columns)),
                size_bytes=uploaded_file.size,
                checksum=file_meta.checksum,
                source_format=file_path.suffix.strip("."),
                description="Uploaded via the AuditHub dashboard.",
                column_names=df.columns.tolist(),
                column_types={c.name: c.dtype for c in summary.column_stats},
            )
            dataset_id = registry.register(meta)
            version = (registry.get(dataset_id).extra_metadata.get("version", 1)
                       if registry.get(dataset_id) else 1)

            cleaning_actions = file_meta.extra_metadata.get("cleaning_actions", [])
            set_dataset(
                df,
                name=uploaded_file.name,
                dataset_id=dataset_id,
                summary=summary,
                cleaning_actions=cleaning_actions,
            )

            total_cells = max(len(df) * len(df.columns), 1)
            missing_pct = df.isna().sum().sum() / total_cells * 100

            ui.divider()
            ui.section("Ingested")

            ui.stat_row([
                {"label": "Rows", "value": f"{len(df):,}"},
                {"label": "Columns", "value": len(df.columns)},
                {"label": "Missing", "value": f"{missing_pct:.1f}%",
                 "sub": ui.badge("clean", "STABLE") if missing_pct == 0
                        else ui.badge("needs repair", "WARNING")},
                {"label": "Detected type",
                 "value": summary.dataset_type.get("type", "unknown").title(),
                 "sub": f"registered as version {version}"},
            ])

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            ui.dataframe(df.head(12))

            if cleaning_actions:
                with st.expander(
                    f"{len(cleaning_actions)} cleaning action(s) applied on load", expanded=False
                ):
                    st.caption(
                        "Applied before analysis, so completeness metrics reflect real "
                        "missing data rather than placeholder text."
                    )
                    ui.dataframe(
                        pd.DataFrame(cleaning_actions)[["step", "column", "detail"]]
                        .fillna({"column": "—"})
                    )
            elif apply_cleaning:
                st.caption("No structural cleaning was required — the file was already tidy.")

            null_counts = df.isna().sum()
            missing_cols = {str(c): int(n) for c, n in null_counts.items() if n > 0}
            if missing_cols:
                listed = ", ".join(f"`{c}` ({n})" for c, n in list(missing_cols.items())[:8])
                st.warning(
                    f"{len(missing_cols)} column(s) contain missing values: {listed}"
                    + (" …" if len(missing_cols) > 8 else "")
                )
            else:
                st.success("No missing values detected.")

        except Exception as exc:
            st.error(f"Could not load this file: {exc}")
            st.caption(
                "For delimited text, check there is a header row. For Excel, confirm "
                "the correct sheet is selected above."
            )

# ---------------------------------------------------------------------------
# Clean and download, without leaving this page
# ---------------------------------------------------------------------------
# The full Repair page gives per-column control, but the most common need is
# simply "clean this and give it back to me". Burying that behind three
# navigation steps made the product look like it had no output.

if has_dataset():
    ui.divider()
    ui.section(
        "Clean it and take it away",
        "One pass: fill every gap, drop duplicate rows, fix column types. "
        "Use Repair if you want to choose the strategy per column.",
    )

    # Measured on the frame as ingested, so the before/after numbers stay
    # meaningful once a repair has replaced the active frame.
    source_df = get_original_df()
    gaps = int(source_df.isna().sum().sum())
    dupes = int(source_df.duplicated().sum())

    action_left, action_right = st.columns([1, 2])
    with action_left:
        target_choice = st.selectbox(
            "Label column (optional)",
            ["None"] + [str(c) for c in source_df.columns],
            help=(
                "The column you would predict. It is never filled in — rows "
                "missing a label are dropped, because inventing labels invents "
                "answers."
            ),
        )
        target = None if target_choice == "None" else target_choice

        if st.button("Clean this dataset", type="primary", width="stretch"):
            with st.spinner("Cleaning…"):
                try:
                    repairer = get_repairer()
                    result = repairer.auto_repair(target_column=target)
                    notify_data_changed()
                    st.session_state["auto_repair_result"] = result.to_dict()
                except Exception as exc:
                    st.error(f"Cleaning failed: {exc}")

    with action_right:
        if has_repairs():
            repairer = get_repairer()
            clean_df = repairer.df
            issues = repairer.verify_clean(target_column=st.session_state.get("target_column"))

            ui.stat_row([
                {"label": "Gaps filled", "value": f"{gaps - int(clean_df.isna().sum().sum()):,}"},
                {"label": "Rows", "value": f"{len(clean_df):,}"},
                {"label": "Status", "value": "Clean" if not issues else "Incomplete",
                 "sub": ui.badge("ready to download", "STABLE") if not issues
                        else ui.badge("see Repair", "WARNING")},
            ])

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            stem = st.session_state.get("dataset_name", "dataset").rsplit(".", 1)[0]
            st.download_button(
                "Download cleaned CSV",
                data=repairer.to_csv_bytes(),
                file_name=f"{stem}_cleaned.csv",
                mime="text/csv",
                type="primary",
                width="stretch",
            )
            st.caption(
                "No missing values, no duplicate rows, no placeholder text, correct "
                "column types. A copy is also saved under `data/repaired/`."
            )
        else:
            st.markdown(
                f'<div class="ah-card" style="padding:18px 20px;">'
                f'<div style="color:var(--text-secondary);font-size:0.9rem;">'
                f"This file has <strong>{gaps:,}</strong> missing value(s) and "
                f"<strong>{dupes:,}</strong> duplicate row(s).</div>"
                f'<div style="color:var(--text-muted);font-size:0.83rem;margin-top:8px;">'
                f"Press <strong>Clean this dataset</strong> to fix them and get a "
                f"download link.</div></div>",
                unsafe_allow_html=True,
            )

ui.divider()
ui.section("Registry", "Every ingested file, newest first.")

registry = DatasetRegistry()
records = registry.list_all()
if not records:
    ui.empty_state("Nothing registered yet.", "Upload a file above to get started.")
else:
    ui.dataframe(
        pd.DataFrame([
            {
                "Dataset": r.filename,
                "Version": r.extra_metadata.get("version", 1),
                "Rows": f"{r.num_rows:,}",
                "Columns": r.num_columns,
                "Format": (r.source_format or "").upper().lstrip("."),
                "Registered": (r.upload_timestamp or "")[:19].replace("T", " "),
            }
            for r in records
        ]),
        height=320,
    )
