"""
AuditHub Page - Mutation Lab
==============================

Deliberately damages a copy of the dataset so downstream defences can be
tested against known corruption.
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
from app.components.session import data_source_caption, require_dataset
from src.mutation_lab.mutator import DatasetMutator

ui.page_setup("Mutation Lab", icon="◑")

ui.page_header(
    "Break it on purpose",
    "Inject missing values, duplicates, outliers, noise or flipped labels into a "
    "copy of the data. Seeded, so every run is reproducible.",
    eyebrow="Step 8 · Mutation",
)

df = require_dataset()
data_source_caption()

MUTATIONS = {
    "Inject missing values": "Blank out a fraction of cells at random.",
    "Inject duplicates": "Repeat a fraction of rows.",
    "Inject outliers": "Push a fraction of numeric values 5–10 standard deviations out.",
    "Inject Gaussian noise": "Add proportional noise to every numeric column.",
    "Shuffle values": "Break the row-wise relationship between columns.",
    "Label flipping": "Change a fraction of labels to a different class.",
    "Random corruption": "Replace a fraction of cells with junk values.",
}

control_column, preview_column = st.columns([1, 2])

with control_column:
    mutation_type = st.selectbox("Mutation", list(MUTATIONS))
    st.caption(MUTATIONS[mutation_type])

    fraction = st.slider("Intensity", 0.0, 0.9, 0.1, 0.05,
                         help="Fraction of rows or cells affected.")
    seed = st.number_input("Seed", min_value=1, max_value=100000, value=42)

    target_col = None
    if mutation_type == "Label flipping":
        target_col = st.selectbox("Label column", df.columns.tolist())

    if st.button("Apply mutation", type="primary", width="stretch"):
        with st.spinner("Injecting anomalies…"):
            try:
                mutator = DatasetMutator(seed=int(seed))
                if mutation_type == "Inject missing values":
                    mutated = mutator.inject_missing_values(df, fraction=fraction)
                elif mutation_type == "Inject duplicates":
                    mutated = mutator.inject_duplicates(df, fraction=fraction)
                elif mutation_type == "Inject outliers":
                    mutated = mutator.inject_outliers(df, fraction=fraction)
                elif mutation_type == "Inject Gaussian noise":
                    mutated = mutator.inject_gaussian_noise(df, noise_level=fraction)
                elif mutation_type == "Shuffle values":
                    mutated = mutator.shuffle_column(df, columns=df.columns.tolist())
                elif mutation_type == "Label flipping":
                    mutated = mutator.flip_labels(df, target_column=target_col, fraction=fraction)
                else:
                    mutated = mutator.inject_random_corruption(df, fraction=fraction)

                st.session_state["mutated_df"] = mutated
            except Exception as exc:
                st.error(f"Mutation failed: {exc}")

with preview_column:
    mutated = st.session_state.get("mutated_df")
    if mutated is None:
        ui.empty_state(
            "No mutation applied yet.",
            "Choose a mutation and press <strong>Apply mutation</strong>.",
        )
    else:
        original_missing = int(df.isna().sum().sum())
        mutated_missing = int(mutated.isna().sum().sum())
        ui.stat_row([
            {"label": "Rows", "value": f"{len(mutated):,}",
             "sub": f"{len(mutated) - len(df):+,} vs original"},
            {"label": "Missing cells", "value": f"{mutated_missing:,}",
             "sub": f"{mutated_missing - original_missing:+,} vs original"},
            {"label": "Duplicates", "value": f"{int(mutated.duplicated().sum()):,}"},
        ])

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            st.caption("Original")
            ui.dataframe(df.head(14))
        with right:
            st.caption("Mutated")
            ui.dataframe(mutated.head(14))

        st.download_button(
            "Download mutated CSV",
            data=mutated.to_csv(index=False).encode("utf-8"),
            file_name="mutated_dataset.csv",
            mime="text/csv",
        )
        st.caption(
            "The mutated copy is kept separate — the active dataset is unchanged. "
            "Use **Robustness** to measure how a model degrades against it."
        )
