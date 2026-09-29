"""
AuditHub Validation - Schema Validator
========================================

Validates the structure and data types of a dataset against an expected schema.
"""

from typing import Any, Dict, List, Optional
import pandas as pd

from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger
from src.validation.exceptions import SchemaValidationError

logger = get_logger(__name__)


class SchemaValidator:
    """Validates dataframe schema and column types against a reference schema.

    Supports validating against configs/schema.yaml or dynamically building
    a schema from a reference DataFrame.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the SchemaValidator.

        Parameters
        ----------
        schema : dict | None
            Reference schema dictionary. If None, loads from ConfigManager.
        """
        if schema is None:
            try:
                self.schema = ConfigManager().get_schema()
            except Exception as exc:
                logger.warning("Could not load schema from ConfigManager: %s. Using empty schema.", exc)
                self.schema = {}
        else:
            self.schema = schema

    @classmethod
    def from_reference_dataframe(cls, df: pd.DataFrame) -> "SchemaValidator":
        """Create a SchemaValidator by inferring the schema from a reference DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Reference DataFrame.

        Returns
        -------
        SchemaValidator
            Initialized validator instance.
        """
        schema = {
            "dataset": {
                "expected_columns": [
                    {
                        "name": str(col),
                        "type": "auto",
                        "nullable": df[col].isna().any(),
                        "unique": df[col].is_unique,
                    }
                    for col in df.columns
                ],
                "constraints": {
                    "min_rows": 1,
                    "max_rows": 100_000_000,
                    "max_missing_fraction": 0.5,
                    "max_duplicate_fraction": 1.0,
                },
                "column_types": {
                    "integer": ["int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8"],
                    "float": ["float64", "float32", "float16"],
                    "categorical": ["object", "category", "bool"],
                    "datetime": ["datetime64[ns]", "datetime64[ns, UTC]"],
                    "text": ["string"],
                }
            }
        }
        return cls(schema=schema)

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a DataFrame against the reference schema.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        dict
            Dictionary containing validation results: success (bool), errors (list), warnings (list).
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated_columns: Dict[str, Dict[str, Any]] = {}

        dataset_schema = self.schema.get("dataset", {})
        expected_columns = dataset_schema.get("expected_columns", [])
        constraints = dataset_schema.get("constraints", {})
        column_types_map = dataset_schema.get("column_types", {})

        # 1. Check Row Constraints
        num_rows = len(df)
        min_rows = constraints.get("min_rows")
        max_rows = constraints.get("max_rows")

        if min_rows is not None and num_rows < min_rows:
            errors.append(f"Row count validation failed: dataset has {num_rows} rows, expected at least {min_rows}.")
        if max_rows is not None and num_rows > max_rows:
            errors.append(f"Row count validation failed: dataset has {num_rows} rows, expected at most {max_rows}.")

        # If there are no expected columns defined, treat current df columns as expected
        if not expected_columns:
            logger.info("No expected columns in schema. Inferring columns from dataframe.")
            expected_columns = [
                {"name": col, "type": "auto", "nullable": True, "unique": False}
                for col in df.columns
            ]

        expected_names = {col_def["name"] for col_def in expected_columns}
        actual_names = set(df.columns)

        # 2. Check Missing and Unexpected Columns
        missing_cols = expected_names - actual_names
        unexpected_cols = actual_names - expected_names

        for col in missing_cols:
            errors.append(f"Missing expected column: '{col}'")

        for col in unexpected_cols:
            warnings.append(f"Unexpected column found in dataset: '{col}'")
            validated_columns[col] = {"status": "warning", "details": "Unexpected column"}

        # 3. Check Column Types and Constraints
        for col_def in expected_columns:
            name = col_def["name"]
            if name not in df.columns:
                validated_columns[name] = {"status": "error", "details": "Column missing"}
                continue

            series = df[name]
            dtype_str = str(series.dtype)
            expected_type = col_def.get("type", "auto")
            nullable = col_def.get("nullable", True)
            unique = col_def.get("unique", False)

            col_errors = []

            # Datatype match
            if expected_type != "auto" and expected_type in column_types_map:
                allowed_dtypes = column_types_map[expected_type]
                # Check direct match or substring match
                is_valid_type = any(
                    allowed in dtype_str or dtype_str in allowed
                    for allowed in allowed_dtypes
                )
                if not is_valid_type:
                    col_errors.append(
                        f"Type mismatch: expected type '{expected_type}' (allowed: {allowed_dtypes}), got '{dtype_str}'"
                    )

            # Nullability check
            if not nullable:
                null_count = series.isna().sum()
                if null_count > 0:
                    col_errors.append(f"Nullability violation: column contains {null_count} null value(s).")

            # Uniqueness check
            if unique:
                if not series.is_unique:
                    col_errors.append("Uniqueness violation: column contains duplicate values.")

            if col_errors:
                errors.extend([f"Column '{name}': {err}" for err in col_errors])
                validated_columns[name] = {"status": "error", "details": col_errors}
            else:
                validated_columns[name] = {"status": "success", "details": "Valid"}

        success = len(errors) == 0

        return {
            "success": success,
            "errors": errors,
            "warnings": warnings,
            "validated_columns": validated_columns,
            "metrics": {
                "num_rows": num_rows,
                "num_columns": len(df.columns),
            }
        }
