"""
AuditHub Repair - Arithmetic Relationship Discovery
====================================================

Finds exact arithmetic relationships between numeric columns and uses them to
*derive* missing values rather than guess them.

Transactional exports are full of columns that are functions of other columns:
``total = quantity * price``, ``net = gross - discount``, ``end = start +
duration``. When one of the three is missing, statistical imputation fills it
with a median -- a plausible number that is simply wrong. If the relationship
holds everywhere else, the true value is computable.

A real cafe-sales export made the cost concrete: 1,514 gaps across quantity,
price and total, of which 1,398 were exactly recoverable. The median filled a
row reading ``4 x 1.00`` with a total of ``8.00`` when the answer was ``4.00``.

A relationship is only used when it holds for effectively every row that could
test it, so a coincidental fit on a handful of rows is never extrapolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Rows that can test a candidate relationship before it is trusted at all.
_MIN_SUPPORT = 30

# Share of testable rows that must satisfy it. Deliberately just under 1.0:
# floating-point money arithmetic can leave a handful of rows off by an ulp,
# but a genuinely different formula fails far more often than this allows.
_MIN_CONFIDENCE = 0.999

# Relative tolerance when comparing the computed value to the recorded one.
_RTOL = 1e-6

# Numeric columns considered. The search is over ordered triples, so a very
# wide frame is capped to keep discovery from dominating a repair run.
_MAX_COLUMNS = 12


@dataclass
class ColumnRelationship:
    """An exact arithmetic identity between three numeric columns.

    Attributes
    ----------
    target : str
        The column on the left of the identity.
    left, right : str
        The two operand columns.
    operator : str
        ``"*"`` or ``"+"``.
    support : int
        Rows the relationship was verified against.
    confidence : float
        Fraction of those rows that satisfied it.
    """

    target: str
    left: str
    right: str
    operator: str
    support: int
    confidence: float

    @property
    def expression(self) -> str:
        """Render the identity, e.g. ``Total = Quantity * Price``."""
        return f"{self.target} = {self.left} {self.operator} {self.right}"

    def compute(self, frame: pd.DataFrame) -> pd.Series:
        """Evaluate the right-hand side over a frame."""
        left = pd.to_numeric(frame[self.left], errors="coerce").astype("float64")
        right = pd.to_numeric(frame[self.right], errors="coerce").astype("float64")
        return left * right if self.operator == "*" else left + right

    def solve_for(self, frame: pd.DataFrame, column: str) -> Optional[pd.Series]:
        """Rearrange the identity to solve for one of the operands.

        Returns ``None`` when ``column`` does not appear in the relationship.
        """
        target = pd.to_numeric(frame[self.target], errors="coerce").astype("float64")

        if column == self.target:
            return self.compute(frame)

        if column in (self.left, self.right):
            other_name = self.right if column == self.left else self.left
            other = pd.to_numeric(frame[other_name], errors="coerce").astype("float64")
            if self.operator == "*":
                # Division by zero cannot recover the operand; leave those gaps.
                return target / other.replace(0, np.nan)
            return target - other

        return None

    def to_dict(self) -> Dict[str, Any]:
        """Return the relationship as a JSON-serialisable dictionary."""
        return {
            "target": self.target,
            "left": self.left,
            "right": self.right,
            "operator": self.operator,
            "expression": self.expression,
            "support": self.support,
            "confidence": round(self.confidence, 6),
        }


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric columns worth considering, most-populated first."""
    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c].dtype)
        and not pd.api.types.is_bool_dtype(df[c].dtype)
    ]
    # An all-null or constant column cannot establish a relationship.
    numeric = [c for c in numeric if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    numeric.sort(key=lambda c: -int(df[c].notna().sum()))
    return numeric[:_MAX_COLUMNS]


def discover_relationships(
    df: pd.DataFrame,
    min_support: int = _MIN_SUPPORT,
    min_confidence: float = _MIN_CONFIDENCE,
) -> List[ColumnRelationship]:
    """Find exact arithmetic identities among the numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to inspect.
    min_support : int
        Minimum rows where all three columns are present.
    min_confidence : float
        Minimum share of those rows that must satisfy the identity.

    Returns
    -------
    list[ColumnRelationship]
        Discovered identities, strongest support first. Empty when the data
        holds none -- which is the common case and costs one pass.
    """
    columns = _numeric_columns(df)
    if len(columns) < 3:
        return []

    numeric = df[columns].apply(pd.to_numeric, errors="coerce").astype("float64")
    found: List[ColumnRelationship] = []
    claimed: set = set()

    for target, left, right in permutations(columns, 3):
        # a * b and a + b are commutative, so only test each unordered pair once.
        if left > right:
            continue
        if target in claimed:
            continue

        testable = numeric[[target, left, right]].notna().all(axis=1)
        support = int(testable.sum())
        if support < min_support:
            continue

        subset = numeric[testable]
        for operator in ("*", "+"):
            computed = (
                subset[left] * subset[right] if operator == "*"
                else subset[left] + subset[right]
            )
            matches = np.isclose(
                computed.values, subset[target].values, rtol=_RTOL, atol=1e-9
            )
            confidence = float(matches.mean())
            if confidence < min_confidence:
                continue

            # A sum that is really "x + 0" or a product that is really "x * 1"
            # is an artefact of a constant column, not a useful identity.
            if subset[left].nunique() < 2 or subset[right].nunique() < 2:
                continue

            found.append(ColumnRelationship(
                target=str(target), left=str(left), right=str(right),
                operator=operator, support=support, confidence=confidence,
            ))
            claimed.add(target)
            logger.info(
                "Discovered relationship %s (%d rows, %.4f confidence)",
                found[-1].expression, support, confidence,
            )
            break

    found.sort(key=lambda r: -r.support)
    return found


def apply_relationships(
    df: pd.DataFrame,
    relationships: List[ColumnRelationship],
    max_passes: int = 3,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Fill gaps that a discovered relationship can compute exactly.

    Runs repeatedly: deriving one column can supply the operand another
    relationship needs, so a second pass often recovers more.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to fill.
    relationships : list[ColumnRelationship]
        Identities from :func:`discover_relationships`.
    max_passes : int
        Cap on the number of fill passes.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        The filled frame and one record per column filled.
    """
    if not relationships:
        return df, []

    out = df.copy()
    filled: Dict[Tuple[str, str], int] = {}

    for _ in range(max_passes):
        changed = 0
        for relationship in relationships:
            for column in (relationship.target, relationship.left, relationship.right):
                if column not in out.columns:
                    continue
                gaps = out[column].isna()
                if not gaps.any():
                    continue

                candidate = relationship.solve_for(out, column)
                if candidate is None:
                    continue

                usable = gaps & candidate.notna() & np.isfinite(candidate)

                is_integer_column = pd.api.types.is_integer_dtype(out[column].dtype)
                if is_integer_column:
                    # Only derive a whole number. Total 8 over price 3 gives
                    # 2.667: rounding that to 3 would invent a value and break
                    # the very identity it came from, so it is left to
                    # imputation instead.
                    #
                    # np.isclose returns an ndarray, so the result is wrapped
                    # back into a Series -- combining it with the boolean mask
                    # directly loses the index alignment.
                    remainder = (candidate % 1).abs().fillna(0.5)
                    whole = pd.Series(
                        np.isclose(remainder.values, 0.0, atol=1e-6)
                        | np.isclose(remainder.values, 1.0, atol=1e-6),
                        index=candidate.index,
                    )
                    usable = usable & whole

                count = int(usable.sum())
                if not count:
                    continue

                values = candidate[usable]
                if is_integer_column:
                    values = values.round().astype("Int64")

                out.loc[usable, column] = values
                key = (column, relationship.expression)
                filled[key] = filled.get(key, 0) + count
                changed += count

        if not changed:
            break

    records = [
        {
            "column": column,
            "expression": expression,
            "cells_filled": count,
            "method": "derived",
        }
        for (column, expression), count in filled.items()
    ]
    if records:
        logger.info(
            "Derived %d cell(s) exactly from %d relationship(s).",
            sum(r["cells_filled"] for r in records), len(relationships),
        )
    return out, records


def check_violations(
    df: pd.DataFrame,
    relationships: List[ColumnRelationship],
) -> List[Dict[str, Any]]:
    """Report rows that break a discovered relationship.

    Separate from filling: a row where all three values are present but the
    arithmetic disagrees is a data-entry error worth surfacing, not a gap.
    """
    findings: List[Dict[str, Any]] = []
    for relationship in relationships:
        columns = [relationship.target, relationship.left, relationship.right]
        if any(c not in df.columns for c in columns):
            continue

        testable = df[columns].notna().all(axis=1)
        if not testable.any():
            continue

        subset = df[testable]
        computed = relationship.compute(subset)
        actual = pd.to_numeric(subset[relationship.target], errors="coerce").astype("float64")
        mismatched = ~np.isclose(computed.values, actual.values, rtol=_RTOL, atol=1e-9)

        count = int(mismatched.sum())
        if count:
            findings.append({
                "expression": relationship.expression,
                "violations": count,
                "checked": int(testable.sum()),
                "example_rows": subset.index[mismatched][:5].tolist(),
            })
    return findings


__all__ = [
    "ColumnRelationship",
    "apply_relationships",
    "check_violations",
    "discover_relationships",
]
