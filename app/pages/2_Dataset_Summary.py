"""
AuditHub Page - Dataset Summary
=================================

Inferred column categories, per-column statistics and detected ML roles.
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
from app.components.session import data_source_caption, get_active_summary, require_dataset

ui.page_setup("Summary", icon="▦")

ui.page_header(
    "What is in this dataset",
    "Column categories, ML roles and per-column statistics, all inferred from the "
    "data itself rather than from column names.",
    eyebrow="Step 2 · Summary",
)

df = require_dataset()
data_source_caption()
summary = get_active_summary()

detected = summary.dataset_type.get("type", "unknown")
confidence = summary.dataset_type.get("confidence", 0.0) * 100
missing_total = summary.missing_summary.get("total_missing", 0)
missing_pct = summary.missing_summary.get("missing_pct", 0.0)

ui.stat_row([
    {"label": "Rows", "value": f"{len(df):,}"},
    {"label": "Columns", "value": len(df.columns)},
    {"label": "Missing cells", "value": f"{missing_total:,}", "sub": f"{missing_pct:.2f}% of all cells"},
    {"label": "Detected type", "value": detected.title(),
     "sub": f"{confidence:.0f}% confidence"},
])

ui.divider()

# ---------------------------------------------------------------------------
# Roles and composition
# ---------------------------------------------------------------------------

left, right = st.columns([1, 1])

with left:
    ui.section("ML roles")

    targets = [str(t.get("column")) for t in summary.potential_targets if t.get("column")]
    ids = [str(i.get("column")) for i in summary.potential_ids if i.get("column")]
    features = [str(f) for f in summary.potential_features if f]

    def _chips(items, empty="none inferred"):
        if not items:
            return f'<span style="color:var(--text-muted);font-size:0.85rem;">{empty}</span>'
        return " ".join(f'<span class="ah-chip">{i}</span>' for i in items[:12])

    ui.card("Target candidates", _chips(targets))
    ui.card("Identifier columns", _chips(ids))
    ui.card(
        f"Predictive features ({len(features)})",
        _chips(features) + (
            '<div style="margin-top:8px;color:var(--text-muted);font-size:0.8rem;">'
            f"+{len(features) - 12} more</div>" if len(features) > 12 else ""
        ),
    )

with right:
    ui.section("Column composition")
    categories: dict = {}
    for column in summary.column_stats:
        categories[column.inferred_type] = categories.get(column.inferred_type, 0) + 1

    if categories:
        composition = pd.DataFrame(
            {"category": list(categories), "columns": list(categories.values())}
        ).sort_values("columns", ascending=True)
        figure = px.bar(
            composition, x="columns", y="category", orientation="h",
            color_discrete_sequence=[ui.PLOT_COLORS["accent"]],
            title="Inferred semantic type",
        )
        ui.plot(figure, height=300, show_legend=False)

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

missing = [
    {"column": c.name, "missing_pct": c.null_pct}
    for c in summary.column_stats if c.null_pct > 0
]
if missing:
    ui.section("Where the gaps are")
    missing_frame = pd.DataFrame(missing).sort_values("missing_pct", ascending=True)
    figure = px.bar(
        missing_frame, x="missing_pct", y="column", orientation="h",
        color_discrete_sequence=[ui.PLOT_COLORS["warn"]],
        labels={"missing_pct": "% missing", "column": ""},
        title="Missing values by column",
    )
    ui.plot(figure, height=max(240, 26 * len(missing_frame)), show_legend=False)

# ---------------------------------------------------------------------------
# Per-column detail
# ---------------------------------------------------------------------------

ui.divider()
ui.section("Column statistics")

ui.dataframe(
    pd.DataFrame([
        {
            "Column": c.name,
            "Type": c.dtype,
            "Inferred": c.inferred_type,
            "Unique": c.nunique,
            "Nulls": c.null_count,
            "Null %": round(c.null_pct, 2),
            "Mean": round(c.mean, 4) if c.mean is not None else None,
            "Std": round(c.std, 4) if c.std is not None else None,
            "Min": round(c.min_val, 4) if c.min_val is not None else None,
            "Max": round(c.max_val, 4) if c.max_val is not None else None,
        }
        for c in summary.column_stats
    ]),
    height=420,
)
