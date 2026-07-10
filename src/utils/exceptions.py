"""
AuditHub - Custom Exception Hierarchy
======================================

Defines a hierarchy of custom exceptions used throughout the AuditHub
platform. All custom exceptions inherit from :class:`AuditHubException`,
which itself inherits from :class:`Exception`.

Exception hierarchy::

    Exception
    └── AuditHubException
        ├── ConfigurationException
        ├── DatasetException
        ├── ValidationException
        ├── ProfilingException
        ├── QualityException
        ├── RepairException
        ├── HealthException
        ├── MutationException
        ├── RobustnessException
        ├── ModelException
        ├── ReportException
        ├── DatabaseException
        ├── PipelineException
        └── IngestionException
"""

from typing import Any, Dict, Optional


class AuditHubException(Exception):
    """Base exception for all AuditHub errors.

    All custom exceptions in the platform should inherit from this class
    to enable consistent error handling and logging.

    Parameters
    ----------
    message : str
        Human-readable error description.
    details : dict | None
        Optional dictionary with additional context about the error.
    cause : BaseException | None
        Optional original exception that caused this error.

    Examples
    --------
    >>> raise AuditHubException("Something went wrong")
    >>> raise AuditHubException("Processing failed", details={"step": "ingestion"})
    """

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        self.cause = cause

        if cause:
            super().__init__(f"{message} [Caused by: {cause}]")
        else:
            super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the exception to a dictionary for logging/reporting.

        Returns
        -------
        dict
            Dictionary with ``error_type``, ``message``, and ``details``.
        """
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationException(AuditHubException):
    """Raised when a configuration error occurs.

    Examples: missing config file, invalid YAML syntax, missing required
    configuration keys, type mismatches in config values.
    """


class DatasetException(AuditHubException):
    """Raised when an error occurs during dataset operations.

    Examples: empty dataset, unsupported format, column mismatch,
    excessive size.
    """


class ValidationException(AuditHubException):
    """Raised when dataset validation fails.

    Examples: schema mismatch, constraint violations, expectation failures
    from Great Expectations.
    """


class ProfilingException(AuditHubException):
    """Raised when dataset profiling fails.

    Examples: profiler crash, unsupported data type, memory exhaustion
    during profiling.
    """


class QualityException(AuditHubException):
    """Raised during quality auditing operations.

    Examples: missing quality dimension, computation error, timeout.
    """


class RepairException(AuditHubException):
    """Raised when data repair operations fail.

    Examples: imputation failure, outlier detection error, repair
    convergence not reached.
    """


class HealthException(AuditHubException):
    """Raised when health score computation fails.

    Examples: missing weight configuration, score out of range,
    aggregation error.
    """


class MutationException(AuditHubException):
    """Raised when data mutation operations fail.

    Examples: mutation parameter out of range, corruption error,
    mutation count exceeded.
    """


class RobustnessException(AuditHubException):
    """Raised when robustness evaluation fails.

    Examples: perturbation error, evaluation metric failure,
    insufficient data for robustness testing.
    """


class ModelException(AuditHubException):
    """Raised when model operations fail.

    Examples: training failure, model load error, inference error,
    incompatible model type.
    """


class ReportException(AuditHubException):
    """Raised when report generation fails.

    Examples: template not found, rendering error, export failure,
    missing data for report.
    """


class DatabaseException(AuditHubException):
    """Raised when database operations fail.

    Examples: connection error, query failure, migration error,
    constraint violation at the database level.
    """


class PipelineException(AuditHubException):
    """Raised when pipeline orchestration fails.

    Examples: stage failure, dependency resolution error, pipeline
    configuration error, execution timeout.
    """


class IngestionException(AuditHubException):
    """Raised when data ingestion fails.

    Examples: file read error, format parsing failure, encoding
    detection failure, file size exceeds limit.
    """
