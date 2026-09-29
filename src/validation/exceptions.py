"""
AuditHub Validation - Exceptions
==================================

Custom exceptions raised by the validation engine.
"""

from src.utils.exceptions import ValidationException


class SchemaValidationError(ValidationException):
    """Raised when schema validation fails (e.g. missing columns, type mismatch)."""
    pass


class ExpectationBuildError(ValidationException):
    """Raised when Great Expectations expectation building fails."""
    pass


class ReportGenerationError(ValidationException):
    """Raised when validation report generation fails."""
    pass
