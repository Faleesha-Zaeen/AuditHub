"""
AuditHub Validation - Expectation Builder
===========================================

Dynamically builds Great Expectations suites based on dataset properties and type inference.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import great_expectations as ge
from great_expectations.dataset.pandas_dataset import PandasDataset

from src.ingestion.dataset_analyzer import ColumnCategory, DatasetSummary
from src.utils.logger import get_logger
from src.validation.exceptions import ExpectationBuildError

logger = get_logger(__name__)


class ExpectationBuilder:
    """Dynamically builds Great Expectations suites from dataset profiling summaries."""

    def __init__(self, summary: DatasetSummary) -> None:
        """Initialize the ExpectationBuilder.

        Parameters
        ----------
        summary : DatasetSummary
            Profiling summary of the dataset from DatasetAnalyzer.
        """
        self.summary = summary

    def build_suite(self, df: pd.DataFrame) -> PandasDataset:
        """Create a Great Expectations PandasDataset and configure its expectation suite.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to build expectations for and validate.

        Returns
        -------
        PandasDataset
            DataFrame wrapped in Great Expectations PandasDataset with configured suite.
        """
        logger.info("Building Great Expectations suite for dataset of shape %s", df.shape)
        try:
            ge_df = ge.from_pandas(df)
        except Exception as exc:
            raise ExpectationBuildError("Failed to initialize Great Expectations dataset from pandas", cause=exc)

        # Iterate over column stats to add expectations dynamically
        for col_info in self.summary.column_stats:
            col_name = col_info.name
            if col_name not in df.columns:
                continue

            # 1. Missing values check (expect_column_values_to_not_be_null)
            # If the column is not supposed to be empty/null, set expectation
            # We allow up to the threshold of nulls, e.g. success_rate = 1.0 - (null_pct / 100)
            if col_info.null_count > 0:
                success_rate = max(0.0, 1.0 - (col_info.null_pct / 100.0))
                ge_df.expect_column_values_to_not_be_null(col_name, mostly=success_rate)
            else:
                ge_df.expect_column_values_to_not_be_null(col_name)

            # 2. Constant columns check
            # We expect a column to have more than 1 unique value if it is not a constant
            if col_info.inferred_type == ColumnCategory.CONSTANT or col_info.nunique <= 1:
                # Expect at least 2 unique values to force it to fail if it's constant,
                # or if we accept it, expect 1 unique value. Let's expect unique values
                # to be between 2 and max rows to flag constant columns.
                ge_df.expect_column_unique_value_count_to_be_between(col_name, min_value=2)
            else:
                ge_df.expect_column_unique_value_count_to_be_between(col_name, min_value=2)

            # 3. Uniqueness check (for ID candidates)
            if col_info.is_id_candidate or col_info.inferred_type == ColumnCategory.IDENTIFIER:
                ge_df.expect_column_values_to_be_unique(col_name)

            # 4. Whitespace-only and empty strings check (for text/string columns)
            if col_info.inferred_type in (ColumnCategory.TEXT, ColumnCategory.CATEGORICAL):
                # Expect column values not to match whitespace-only regex
                ge_df.expect_column_values_to_not_match_regex(col_name, r"^\s+$")

            # 5. Type expectations
            # Map semantic type to GE type checks
            if col_info.is_numeric:
                ge_df.expect_column_values_to_be_in_type_list(
                    col_name,
                    ["int", "int64", "int32", "float", "float64", "float32", "number"]
                )
                # Numeric bounds check
                if col_info.min_val is not None and col_info.max_val is not None:
                    # Allow values slightly outside min/max range (e.g. 3 standard deviations or standard buffer)
                    std_val = col_info.std or 0.0
                    buffer = 3.0 * std_val if std_val > 0 else 1.0
                    min_bound = col_info.min_val - buffer
                    max_bound = col_info.max_val + buffer
                    ge_df.expect_column_values_to_be_between(
                        col_name,
                        min_value=float(min_bound),
                        max_value=float(max_bound)
                    )

                # Low variance check
                if col_info.std is not None:
                    # Expect standard deviation to be at least 0.0001 (excluding constant columns)
                    if col_info.std > 0:
                        ge_df.expect_column_stdev_to_be_between(col_name, min_value=1e-4)

            elif col_info.inferred_type == ColumnCategory.BOOLEAN:
                ge_df.expect_column_values_to_be_in_set(col_name, [True, False, 1, 0, "True", "False", "1", "0"])

            elif col_info.inferred_type == ColumnCategory.DATETIME:
                # Expect values to match ISO datetime format or parse cleanly
                # Using a generic type check
                ge_df.expect_column_values_to_be_in_type_list(
                    col_name,
                    ["datetime64[ns]", "datetime64[ns, UTC]", "datetime", "date", "str", "object"]
                )

            elif col_info.inferred_type == ColumnCategory.CATEGORICAL:
                # If unique values are relatively few, check set membership
                if col_info.nunique > 0 and col_info.nunique <= 20:
                    try:
                        unique_values = df[col_name].dropna().unique().tolist()
                        ge_df.expect_column_values_to_be_in_set(col_name, unique_values)
                    except Exception:
                        pass

        # Overall dataset constraints
        # Ensure row count is not 0
        ge_df.expect_table_row_count_to_be_between(min_value=1)
        # Ensure columns match expectations
        ge_df.expect_table_columns_to_match_ordered_list(df.columns.tolist())

        logger.debug("Expectation suite successfully configured.")
        return ge_df
