"""
AuditHub - Pipeline Orchestration Module
=========================================

Coordinates the end-to-end flow:

    Ingestion -> Validation -> Profiling -> Quality -> Health
              -> Repair -> Health (again) -> Training -> Evaluation -> Reporting

Run it headlessly from the command line::

    python -m src.pipeline run data/raw/sales.csv --target revenue

or programmatically::

    from src.pipeline import PipelineContext, create_default_pipeline

    pipeline = create_default_pipeline()
    context = pipeline.run(PipelineContext(dataset_path=Path("data.csv")))
    print(pipeline.status, context.artifacts)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.pipeline.core import (
    Pipeline,
    PipelineContext,
    PipelineStage,
    PipelineStatus,
    StageResult,
    StageStatus,
)
from src.pipeline.stages import (
    DriftStage,
    EvaluationStage,
    ExplainabilityStage,
    HealthStage,
    IngestionStage,
    ProfilingStage,
    QualityStage,
    RepairStage,
    ReportingStage,
    TrainingStage,
    ValidationStage,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Default Pipeline Factory
# ============================================================================


def create_default_pipeline() -> Pipeline:
    """Create the default AuditHub pipeline with all stages.

    Health is measured twice -- before and after repair -- so a run reports the
    improvement rather than a single score.

    Returns
    -------
    Pipeline
        Pipeline instance with all stages registered in order.

    Examples
    --------
    >>> pipeline = create_default_pipeline()
    >>> context = pipeline.run(PipelineContext(dataset_path=Path("data.csv")))
    """
    return Pipeline(
        name="default",
        stages=[
            IngestionStage(),
            ValidationStage(),
            # Drift sits beside validation: both ask whether the incoming data
            # honours an expected contract. It skips itself unless a reference
            # dataset was supplied, so the existing flow is unchanged.
            DriftStage(),
            ProfilingStage(),
            QualityStage(),
            HealthStage("health"),
            RepairStage(),
            HealthStage("health_after"),
            TrainingStage(),
            EvaluationStage(),
            ExplainabilityStage(),
            ReportingStage(),
        ],
    )


def run_pipeline(
    dataset_path: Union[str, Path],
    target_column: Optional[str] = None,
    **options: Any,
) -> tuple:
    """Run the default pipeline over a dataset file.

    Parameters
    ----------
    dataset_path : str | Path
        Dataset to process.
    target_column : str | None
        Label column. Inferred from the data when omitted.
    **options
        Pipeline options, e.g. ``profile=False``, ``train=False``,
        ``numeric_strategy="mean"``, ``report_format="json"``,
        ``reference_path="data/raw/train.csv"`` to enable drift detection.

    Returns
    -------
    tuple[Pipeline, PipelineContext]
        The pipeline (carrying the final status) and the populated context.
    """
    pipeline = create_default_pipeline()
    context = PipelineContext(
        dataset_path=Path(dataset_path),
        dataset_name=Path(dataset_path).name,
        target_column=target_column,
        config=dict(options),
    )

    if options.get("lineage", True):
        try:
            from src.lineage.tracker import LineageTracker

            context.lineage = LineageTracker(dataset_name=Path(dataset_path).name)
        except Exception as exc:  # tracking is optional, never fatal
            logger.warning("Lineage tracking unavailable: %s", exc)

    context = pipeline.run(context)
    return pipeline, context


__all__ = [
    "Pipeline",
    "PipelineContext",
    "PipelineStage",
    "PipelineStatus",
    "StageResult",
    "StageStatus",
    "create_default_pipeline",
    "run_pipeline",
    # Stages
    "DriftStage",
    "IngestionStage",
    "ValidationStage",
    "ProfilingStage",
    "QualityStage",
    "HealthStage",
    "RepairStage",
    "TrainingStage",
    "EvaluationStage",
    "ExplainabilityStage",
    "ReportingStage",
]
