"""
AuditHub - Shared Session State
================================

Single source of truth for "which DataFrame is the app currently working on".

Every analysis page used to read ``st.session_state["df"]`` directly, which is
the *originally uploaded* frame. Repairs were held inside the ``DatasetRepairer``
kept on the Repair page, so imputing a column changed nothing anywhere else --
the Quality Audit still reported the same missing percentages and the exported
report still described the broken data.

Pages now call :func:`get_active_df`, which returns the repaired frame once any
repair has been applied and the original otherwise.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Session keys derived from the loaded dataset. Cleared whenever a new file is
# ingested so results from a previous dataset can never be shown against it.
_DERIVED_KEYS: Tuple[str, ...] = (
    "repairer",
    "summary",
    "validation_report",
    "profiling_summary",
    "quality_report",
    "health_report",
    "health_report_before",
    "robustness_results",
    "training_results",
    "mutated_df",
    "cleaning_actions",
    "target_column",
    "auto_repair_result",
    "_active_summary",
    "_active_summary_key",
)


def has_dataset() -> bool:
    """Return True when a dataset has been ingested this session."""
    return "df" in st.session_state and st.session_state["df"] is not None


def require_dataset(message: str = "Please upload a dataset first on the Upload page.") -> pd.DataFrame:
    """Return the active DataFrame, or halt the page with a warning.

    Parameters
    ----------
    message : str
        Warning shown when no dataset has been ingested.

    Returns
    -------
    pd.DataFrame
        The active (repaired, if any) DataFrame.
    """
    if not has_dataset():
        st.warning(message)
        st.stop()
    return get_active_df()


def set_dataset(
    df: pd.DataFrame,
    name: str,
    dataset_id: Optional[str] = None,
    summary: Any = None,
    cleaning_actions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Install a freshly ingested dataset as the active one.

    Clears every derived artifact so a new upload never inherits the previous
    dataset's reports, repairer or trained model.
    """
    for key in _DERIVED_KEYS:
        st.session_state.pop(key, None)

    st.session_state["df"] = df
    st.session_state["dataset_name"] = name
    if dataset_id is not None:
        st.session_state["dataset_id"] = dataset_id
    if summary is not None:
        st.session_state["summary"] = summary
    st.session_state["cleaning_actions"] = cleaning_actions or []


def get_original_df() -> pd.DataFrame:
    """Return the dataset exactly as it was ingested, before any repair."""
    return st.session_state["df"]


def get_active_df() -> pd.DataFrame:
    """Return the frame every analysis page should work on.

    This is the repaired frame when repairs have been applied, otherwise the
    originally ingested one.
    """
    repairer = st.session_state.get("repairer")
    if repairer is not None and repairer.get_log():
        return repairer.df
    return st.session_state["df"]


def get_active_summary():
    """Return a :class:`DatasetSummary` describing the *active* frame.

    The summary computed at upload describes the original data. Once repairs
    are applied it is stale -- it would still report the missing percentages
    the user just fixed -- so it is recomputed and cached against the current
    repair step.
    """
    from src.ingestion.dataset_analyzer import DatasetAnalyzer

    if not has_repairs():
        summary = st.session_state.get("summary")
        if summary is not None:
            return summary
        # No summary cached (e.g. the dataset was restored into the session
        # without going through the Upload page) -- compute one rather than
        # letting the page fail on a None.
        cache_key = (0, st.session_state["df"].shape)
    else:
        repairer = st.session_state["repairer"]
        cache_key = (len(repairer.get_log()), repairer.df.shape)

    if st.session_state.get("_active_summary_key") != cache_key:
        st.session_state["_active_summary"] = DatasetAnalyzer().analyze(
            get_active_df(),
            source_filename=st.session_state.get("dataset_name"),
        )
        st.session_state["_active_summary_key"] = cache_key
    return st.session_state["_active_summary"]


def has_repairs() -> bool:
    """Return True when at least one repair has been applied."""
    repairer = st.session_state.get("repairer")
    return bool(repairer is not None and repairer.get_log())


def get_repairer():
    """Return the session's :class:`DatasetRepairer`, creating it on demand."""
    from src.repair.repairer import DatasetRepairer

    if "repairer" not in st.session_state:
        st.session_state["repairer"] = DatasetRepairer(st.session_state["df"])
    return st.session_state["repairer"]


def reset_repairer() -> None:
    """Discard all repairs and start again from the ingested dataset."""
    st.session_state.pop("repairer", None)
    st.session_state.pop("auto_repair_result", None)
    _invalidate_reports()


def _invalidate_reports() -> None:
    """Drop cached reports that describe a now-superseded version of the data."""
    for key in (
        "validation_report",
        "profiling_summary",
        "quality_report",
        "health_report",
        "robustness_results",
        "_active_summary",
        "_active_summary_key",
    ):
        st.session_state.pop(key, None)


def notify_data_changed() -> None:
    """Signal that the active frame changed, so stale reports are not shown.

    Called after a repair is applied. Without this, a Quality Audit run before
    the repair would linger on screen and appear to contradict the fix.
    """
    _invalidate_reports()


def get_target_column() -> Optional[str]:
    """Return the target column chosen anywhere in the app, if any."""
    return st.session_state.get("target_column")


def set_target_column(target: Optional[str]) -> None:
    """Remember the chosen target column across pages."""
    if target:
        st.session_state["target_column"] = target
    else:
        st.session_state.pop("target_column", None)


def target_selector(
    df: pd.DataFrame,
    label: str = "Target column",
    help_text: Optional[str] = None,
) -> Optional[str]:
    """Render a target-column selectbox that remembers the choice across pages.

    Returns
    -------
    str | None
        The selected target column, or ``None`` for "not set".
    """
    options = ["None"] + [str(c) for c in df.columns]
    remembered = get_target_column()
    index = options.index(remembered) if remembered in options else 0

    choice = st.selectbox(label, options, index=index, help=help_text)
    target = None if choice == "None" else choice
    set_target_column(target)
    return target


def data_source_caption() -> None:
    """Show which version of the data the current page is analysing."""
    if has_repairs():
        repairer = st.session_state["repairer"]
        st.caption(
            f"Analysing the **repaired** dataset "
            f"({len(repairer.get_log())} repair step(s) applied). "
            "Reset on the Repair page to work from the original."
        )
    else:
        st.caption("Analysing the **originally uploaded** dataset (no repairs applied yet).")


__all__ = [
    "data_source_caption",
    "get_active_df",
    "get_original_df",
    "get_repairer",
    "get_target_column",
    "has_dataset",
    "has_repairs",
    "notify_data_changed",
    "require_dataset",
    "reset_repairer",
    "set_dataset",
    "set_target_column",
    "target_selector",
]
