"""
AuditHub Ingestion - Text Column Roles
=======================================

Decides what *kind* of text a column holds, because the right way to clean a
column depends entirely on that.

Three roles behave differently:

``categorical``
    A small vocabulary repeated many times -- a payment method, a city, a
    status. Mode imputation is defensible here: the most common value is a
    genuine estimate.

``free_text``
    Sentences written by a person -- a review, a comment, a log line. Neither
    sentinel matching nor mode imputation is safe. Copying the most common
    review into an empty one fabricates testimony that nobody gave.

``identifier``
    Values that are nearly all distinct -- an email, an order reference, a
    name. The modal value is not an estimate of a missing one, it is just some
    other row's data.

The distinction also protects real words. ``"None"`` in an allergy column means
the patient has no allergies; ``"Unknown"`` is a real answer on a census form;
``Null`` is a surname. Those must survive, while an upstream exporter's
``"UNKNOWN"`` placeholder must not.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# A column is free text once its values stop looking like labels: several words
# on average, or simply long. Both thresholds are deliberately generous -- a
# two-word category like "Digital Wallet" or "In-store" must stay categorical.
_FREE_TEXT_MIN_WORDS = 3.5
_FREE_TEXT_MIN_CHARS = 40

# Above this share of distinct values a column identifies rows rather than
# grouping them, so no single value is a sensible fill for another.
_IDENTIFIER_UNIQUE_RATIO = 0.9

# Below this many rows the ratios above are noise, so only length is trusted.
_MIN_ROWS_FOR_RATIO = 20

_WORD_RE = re.compile(r"\S+")


def _sample(series: pd.Series, limit: int = 2000) -> pd.Series:
    """Return non-null values as strings, capped for speed on large frames."""
    values = series.dropna()
    if len(values) > limit:
        values = values.iloc[:limit]
    return values.astype(str)


def classify_text_column(series: pd.Series) -> str:
    """Return the role of a column: ``categorical``, ``free_text``,
    ``identifier`` or ``non_text``.

    Parameters
    ----------
    series : pd.Series
        Column to classify.

    Returns
    -------
    str
        One of the four role names. Anything that is not an object/string
        column is ``"non_text"`` and is left to the numeric and datetime paths.
    """
    if series.dtype != object and not isinstance(series.dtype, pd.StringDtype):
        return "non_text"

    values = _sample(series)
    if values.empty:
        return "categorical"

    lengths = values.str.len()
    words = values.map(lambda v: len(_WORD_RE.findall(v)))

    if words.mean() >= _FREE_TEXT_MIN_WORDS or lengths.mean() >= _FREE_TEXT_MIN_CHARS:
        return "free_text"

    non_null = int(series.notna().sum())
    if non_null >= _MIN_ROWS_FOR_RATIO:
        ratio = series.nunique(dropna=True) / non_null
        if ratio >= _IDENTIFIER_UNIQUE_RATIO:
            return "identifier"

    return "categorical"


def is_free_text(series: pd.Series) -> bool:
    """True when a column holds prose rather than labels."""
    return classify_text_column(series) == "free_text"


def is_imputable_text(series: pd.Series) -> bool:
    """True when copying a value from another row is a defensible estimate.

    False for prose and for identifiers, where the modal value is not an
    estimate of the missing one -- it is simply a different row's data.
    """
    return classify_text_column(series) == "categorical"


def casing_class(value: str) -> str:
    """Classify a string's capitalisation: ``upper``, ``lower``, ``title``,
    ``mixed`` or ``none`` (no letters at all).

    Capitalisation is the tell that separates a machine's placeholder from a
    person's answer. An exporter writes ``UNKNOWN``; a form designer writes
    ``Unknown``.
    """
    if not any(ch.isalpha() for ch in value):
        return "none"
    if value.isupper():
        return "upper"
    if value.islower():
        return "lower"
    if value.istitle():
        return "title"
    return "mixed"


def dominant_casing(values: pd.Series, ignore: Optional[set] = None) -> Optional[str]:
    """Return the capitalisation convention a column follows, if it has one.

    Parameters
    ----------
    values : pd.Series
        The column's values.
    ignore : set[str] | None
        Lowercased tokens to exclude, so that candidate placeholders do not
        vote on the convention they are being judged against.

    Returns
    -------
    str | None
        The most common casing class, or ``None`` when too few values carry
        letters to establish one.
    """
    ignore = ignore or set()
    sample = _sample(values)
    if sample.empty:
        return None

    considered = sample[~sample.str.strip().str.lower().isin(ignore)]
    if considered.empty:
        return None

    classes = considered.map(casing_class)
    classes = classes[classes != "none"]
    if classes.empty:
        return None

    return str(classes.mode().iloc[0])


__all__ = [
    "casing_class",
    "classify_text_column",
    "dominant_casing",
    "is_free_text",
    "is_imputable_text",
]
