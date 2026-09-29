"""
AuditHub - Pipeline Core Types
===============================

Status enums, the shared context, the stage interface, and the orchestrator.

The orchestrator's contract: a stage that fails is recorded as ``FAILED`` and,
if it is marked ``critical``, aborts the run. A pipeline that could not do what
it was asked never reports ``COMPLETED`` -- the previous skeleton swallowed
``NotImplementedError`` and returned ``COMPLETED`` having executed nothing.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Pipeline Status
# ============================================================================


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


class PipelineStatus(Enum):
    """Overall status of a pipeline run."""

    NOT_STARTED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIALLY_COMPLETED = auto()


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class StageResult:
    """Result of a single pipeline stage execution.

    Attributes
    ----------
    stage_name : str
        Name of the stage.
    status : StageStatus
        Execution status.
    output : dict | None
        Output data produced by the stage.
    error : str | None
        Error message if the stage failed.
    duration_seconds : float | None
        Execution duration in seconds.
    """

    stage_name: str
    status: StageStatus = StageStatus.PENDING
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    @property
    def ok(self) -> bool:
        """True when the stage completed successfully."""
        return self.status is StageStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as a plain dictionary."""
        return {
            "stage": self.stage_name,
            "status": self.status.name,
            "output": self.output,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class PipelineContext:
    """Shared context passed between pipeline stages.

    Attributes
    ----------
    dataset_path : Path | None
        Path to the current dataset.
    dataset_id : str | None
        Unique identifier for the dataset.
    dataset_name : str | None
        Human-readable dataset name, used for artifact filenames.
    target_column : str | None
        Label column, when the dataset has one.
    df : pd.DataFrame | None
        The active DataFrame. Stages that transform the data replace it, so
        later stages automatically see repaired data.
    original_df : pd.DataFrame | None
        The DataFrame as first ingested, kept for before/after comparison.
    config : dict
        Pipeline configuration.
    metadata : dict
        Shared metadata accumulated across stages.
    results : dict
        Results from completed stages.
    artifacts : dict
        Paths to files written by stages.
    lineage : LineageTracker | None
        Recorder for what each stage did. When ``None`` nothing is tracked,
        so lineage stays optional and cannot break a run.
    """

    dataset_path: Optional[Path] = None
    dataset_id: Optional[str] = None
    dataset_name: Optional[str] = None
    target_column: Optional[str] = None
    df: Optional[pd.DataFrame] = None
    original_df: Optional[pd.DataFrame] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, StageResult] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    lineage: Optional[Any] = None

    def record(
        self,
        stage: str,
        summary: str,
        event_type: str = "",
        column: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a lineage event, if lineage tracking is enabled."""
        if self.lineage is None:
            return
        self.lineage.record(
            stage=stage, summary=summary, event_type=event_type,
            column=column, detail=detail,
        )

    def require_df(self, stage_name: str) -> pd.DataFrame:
        """Return the active DataFrame or raise a clear error.

        Raises
        ------
        RuntimeError
            If no DataFrame has been ingested yet.
        """
        if self.df is None:
            raise RuntimeError(
                f"Stage '{stage_name}' needs a dataset, but ingestion has not produced one."
            )
        return self.df

    def option(self, key: str, default: Any = None) -> Any:
        """Read a configuration option."""
        return self.config.get(key, default)


# ============================================================================
# Abstract Pipeline Stage
# ============================================================================


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages.

    Subclasses must implement :meth:`execute`.

    Parameters
    ----------
    name : str
        Stage name, used as the key in ``PipelineContext.results``.
    critical : bool
        When True, a failure aborts the whole run. Ingestion is critical --
        nothing downstream can work without data -- while profiling is not.
    """

    critical: bool = False

    def __init__(self, name: str, critical: Optional[bool] = None) -> None:
        self.name = name
        if critical is not None:
            self.critical = critical
        self.logger = get_logger(f"audithub.pipeline.{name}")

    @abstractmethod
    def execute(self, context: PipelineContext) -> StageResult:
        """Execute this pipeline stage.

        Parameters
        ----------
        context : PipelineContext
            Shared context with data from previous stages.

        Returns
        -------
        StageResult
            Result of the stage execution.
        """
        ...

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        """Return a reason to skip this stage, or ``None`` to run it.

        Lets a stage opt out cleanly -- training without a target column is
        skipped with an explanation rather than raising.
        """
        return None

    def _result(
        self,
        status: StageStatus = StageStatus.COMPLETED,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> StageResult:
        """Build a :class:`StageResult` for this stage."""
        return StageResult(stage_name=self.name, status=status, output=output, error=error)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# ============================================================================
# Pipeline Orchestrator
# ============================================================================


class Pipeline:
    """Orchestrates the execution of a sequence of pipeline stages.

    Parameters
    ----------
    name : str
        Pipeline name.
    stages : list[PipelineStage] | None
        Ordered list of stages to execute.
    """

    def __init__(
        self,
        name: str = "default",
        stages: Optional[List[PipelineStage]] = None,
    ) -> None:
        self.name = name
        self.stages: List[PipelineStage] = stages or []
        self.logger = get_logger(f"audithub.pipeline.{name}")
        self._status: PipelineStatus = PipelineStatus.NOT_STARTED

    @property
    def status(self) -> PipelineStatus:
        """Get the current pipeline status."""
        return self._status

    def add_stage(self, stage: PipelineStage) -> None:
        """Append a stage to the pipeline.

        Parameters
        ----------
        stage : PipelineStage
            Stage to add.
        """
        self.stages.append(stage)
        self.logger.debug("Added stage: %s", stage.name)

    def run(self, context: Optional[PipelineContext] = None) -> PipelineContext:
        """Execute all pipeline stages in order.

        Parameters
        ----------
        context : PipelineContext | None
            Initial context to use. Creates a new one if not provided.

        Returns
        -------
        PipelineContext
            Final context with results from all executed stages.
        """
        ctx = context or PipelineContext()
        self._status = PipelineStatus.RUNNING

        failed: List[str] = []
        aborted = False

        for stage in self.stages:
            skip_reason = stage.should_skip(ctx)
            if skip_reason:
                self.logger.info("Skipping stage '%s': %s", stage.name, skip_reason)
                ctx.results[stage.name] = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.SKIPPED,
                    output={"reason": skip_reason},
                )
                continue

            self.logger.info("Running stage: %s", stage.name)
            started = time.perf_counter()
            try:
                result = stage.execute(ctx)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                result = StageResult(
                    stage_name=stage.name,
                    status=StageStatus.FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.logger.exception("Stage '%s' raised", stage.name)

            result.duration_seconds = round(time.perf_counter() - started, 3)
            ctx.results[stage.name] = result

            if result.status is StageStatus.FAILED:
                failed.append(stage.name)
                self.logger.error("Stage '%s' failed: %s", stage.name, result.error)
                if stage.critical:
                    self.logger.error("Stage '%s' is critical; aborting run.", stage.name)
                    aborted = True
                    break
            else:
                self.logger.info(
                    "Stage '%s' completed in %.3fs", stage.name, result.duration_seconds
                )

        executed = [r for r in ctx.results.values() if r.status is not StageStatus.SKIPPED]
        if aborted or (failed and not any(r.ok for r in executed)):
            self._status = PipelineStatus.FAILED
        elif failed:
            self._status = PipelineStatus.PARTIALLY_COMPLETED
        else:
            self._status = PipelineStatus.COMPLETED

        self.logger.info(
            "Pipeline '%s' finished with status %s (%d/%d stages completed).",
            self.name, self._status.name,
            sum(1 for r in ctx.results.values() if r.ok), len(self.stages),
        )
        return ctx

    def summary_rows(self, context: PipelineContext) -> List[Dict[str, Any]]:
        """Return a per-stage summary suitable for tabular display."""
        rows = []
        for stage in self.stages:
            result = context.results.get(stage.name)
            rows.append({
                "stage": stage.name,
                "status": result.status.name if result else "NOT_RUN",
                "seconds": result.duration_seconds if result else None,
                "detail": (result.error if result and result.error else _brief(result)),
            })
        return rows

    def __repr__(self) -> str:
        return (
            f"Pipeline(name='{self.name}', "
            f"stages={len(self.stages)}, "
            f"status={self._status})"
        )


def _brief(result: Optional[StageResult]) -> str:
    """Return a short one-line description of a stage's output."""
    if result is None or not result.output:
        return ""
    parts = []
    for key, value in result.output.items():
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
        if len(parts) >= 3:
            break
    return ", ".join(parts)


__all__ = [
    "Pipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineStatus",
    "StageResult",
    "StageStatus",
]
