"""
AuditHub - Pipeline Stages
===========================

Concrete implementations of the nine pipeline stages, wiring the existing
engines together into one headless run:

    Ingestion -> Validation -> Profiling -> Quality -> Health
              -> Repair -> Training -> Evaluation -> Reporting

Each stage reads what it needs from the :class:`PipelineContext` and writes its
findings back, so the Repair stage's output is what Training and Reporting
consume. Stages that cannot run (no target column, for example) skip with a
stated reason rather than failing the run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.pipeline.core import PipelineContext, PipelineStage, StageResult, StageStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Ingestion
# ============================================================================


class IngestionStage(PipelineStage):
    """Load the dataset from disk, clean it structurally, and register it."""

    critical = True

    def __init__(self) -> None:
        super().__init__("ingestion")

    def execute(self, context: PipelineContext) -> StageResult:
        from src.ingestion.dataset_analyzer import DatasetAnalyzer
        from src.ingestion.dataset_loader import DatasetLoader

        if context.dataset_path is None:
            return self._result(
                StageStatus.FAILED,
                error="No dataset_path was provided to the pipeline context.",
            )

        path = Path(context.dataset_path)
        loader = DatasetLoader()
        df, metadata = loader.load(
            path,
            clean=context.option("clean", True),
            max_rows=context.option("max_rows"),
        )

        context.df = df
        context.original_df = df.copy()
        context.dataset_name = context.dataset_name or path.name
        context.metadata["file_metadata"] = metadata.to_dict()

        cleaning_actions = metadata.extra_metadata.get("cleaning_actions", [])
        context.metadata["cleaning_actions"] = cleaning_actions

        summary = DatasetAnalyzer().analyze(df, source_filename=path.name)
        context.metadata["summary"] = summary

        # Infer the target column when the caller did not name one.
        if not context.target_column and summary.potential_targets:
            inferred = summary.potential_targets[0].get("column")
            if inferred:
                context.target_column = str(inferred)
                self.logger.info("Inferred target column: %s", context.target_column)

        if context.option("register", True):
            try:
                from src.utils.dataset_registry import DatasetRegistry

                context.dataset_id = DatasetRegistry().register(
                    metadata, version_group=context.option("version_group")
                )
                if context.lineage is not None:
                    context.lineage.dataset_id = context.dataset_id
                    context.lineage.dataset_name = context.dataset_name or ""
            except Exception as exc:  # registry is a convenience, not a blocker
                self.logger.warning("Could not register dataset: %s", exc)

        # Lineage: the dataset's starting condition, per column.
        context.record(
            self.name,
            f"Loaded {len(df):,} rows x {len(df.columns)} columns from {path.name}",
            event_type="load",
            detail={"rows": int(len(df)), "columns": int(len(df.columns))},
        )
        for action in cleaning_actions:
            context.record(
                self.name, action.get("detail", ""), event_type=action.get("step", "clean"),
                column=action.get("column"), detail=action,
            )
        for col in df.columns:
            nulls = int(df[col].isna().sum())
            if nulls:
                context.record(
                    self.name,
                    f"{nulls} missing value(s) detected ({nulls / len(df) * 100:.1f}%)",
                    event_type="missing_detected",
                    column=str(col),
                    detail={"missing": nulls, "rows": int(len(df))},
                )

        return self._result(output={
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing_cells": int(df.isna().sum().sum()),
            "cleaning_actions": len(cleaning_actions),
            "target_column": context.target_column,
            "dataset_id": context.dataset_id,
        })


# ============================================================================
# Validation
# ============================================================================


class ValidationStage(PipelineStage):
    """Run schema, Great Expectations and custom validation checks."""

    def __init__(self) -> None:
        super().__init__("validation")

    def execute(self, context: PipelineContext) -> StageResult:
        from src.validation.validator import DatasetValidator

        df = context.require_df(self.name)
        report = DatasetValidator().validate(
            df,
            target_column=context.target_column,
            dataset_name=context.dataset_name,
        )
        context.metadata["validation_report"] = report.to_dict()

        context.record(
            self.name,
            f"Validation score {report.score:.1f}/100 "
            f"({'passed' if report.success else 'failed'})",
            event_type="validated",
            detail={"score": float(report.score), "success": bool(report.success)},
        )

        return self._result(output={
            "score": round(float(report.score), 2),
            "success": bool(report.success),
        })


# ============================================================================
# Profiling
# ============================================================================


class ProfilingStage(PipelineStage):
    """Generate a ydata-profiling HTML report."""

    def __init__(self) -> None:
        super().__init__("profiling")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("profile", True):
            return "profiling disabled for this run"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.profiling.profiler import DatasetProfiler

        df = context.require_df(self.name)
        profiler = DatasetProfiler()
        # Full profiling is very slow on wide frames; minimal mode keeps a
        # headless run practical.
        if len(df.columns) > context.option("profiling_full_max_columns", 30):
            profiler.minimal = True

        summary = profiler.profile(df, dataset_name=context.dataset_name)
        context.metadata["profiling_summary"] = summary

        report_path = summary.get("report_path") if isinstance(summary, dict) else None
        if report_path:
            context.artifacts["profile_report"] = str(report_path)

        return self._result(output={"report_path": report_path})


# ============================================================================
# Quality
# ============================================================================


class QualityStage(PipelineStage):
    """Run the rule-based quality audit."""

    def __init__(self) -> None:
        super().__init__("quality")

    def execute(self, context: PipelineContext) -> StageResult:
        from src.quality.auditor import DatasetAuditor

        df = context.require_df(self.name)
        report = DatasetAuditor().audit(df, target_column=context.target_column)
        context.metadata["quality_report"] = report.to_dict()

        counts = report.summary
        context.record(
            self.name,
            f"{counts.get('ERROR', 0)} error(s), {counts.get('WARNING', 0)} warning(s), "
            f"{counts.get('INFO', 0)} note(s)",
            event_type="audited", detail=dict(counts),
        )
        for finding in report.findings:
            if finding.column and finding.severity in ("ERROR", "WARNING"):
                context.record(
                    self.name, f"{finding.severity.lower()}: {finding.message}",
                    event_type="finding", column=str(finding.column),
                    detail={"severity": finding.severity, "dimension": finding.dimension},
                )
        return self._result(output={
            "errors": int(counts.get("ERROR", 0)),
            "warnings": int(counts.get("WARNING", 0)),
            "info": int(counts.get("INFO", 0)),
        })


# ============================================================================
# Health
# ============================================================================


class HealthStage(PipelineStage):
    """Compute the composite health score and grade.

    Scheduled twice in the default pipeline -- once before repair and once
    after -- so the run reports the improvement rather than a single number.
    """

    def __init__(self, name: str = "health") -> None:
        super().__init__(name)

    def execute(self, context: PipelineContext) -> StageResult:
        from src.health.calculator import HealthScoreCalculator

        df = context.require_df(self.name)
        report = HealthScoreCalculator().calculate(df, target_column=context.target_column)

        # The first health run is the "before" baseline; a later run (after
        # repair) is recorded separately so the report can show the delta.
        key = "health_report_after" if "health_report" in context.metadata else "health_report"
        context.metadata[key] = report.to_dict()

        context.record(
            self.name,
            f"Health {report.grade} ({report.overall_score:.2f}/100)",
            event_type="scored",
            detail={"score": float(report.overall_score), "grade": report.grade},
        )

        output: Dict[str, Any] = {
            "score": round(float(report.overall_score), 2),
            "grade": report.grade,
        }
        before = context.metadata.get("health_report")
        if key == "health_report_after" and before:
            output["improvement"] = round(
                float(report.overall_score) - float(before.get("overall_score", 0.0)), 2
            )
            output["was"] = f"{before.get('grade')} ({before.get('overall_score')})"

        return self._result(output=output)


# ============================================================================
# Repair
# ============================================================================


class RepairStage(PipelineStage):
    """Auto-repair the dataset and export the cleaned copy."""

    def __init__(self) -> None:
        super().__init__("repair")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("repair", True):
            return "repair disabled for this run"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.repair.repairer import DatasetRepairer

        df = context.require_df(self.name)
        repairer = DatasetRepairer(df)
        result = repairer.auto_repair(
            target_column=context.target_column,
            numeric_strategy=context.option("numeric_strategy", "median"),
            categorical_strategy=context.option("categorical_strategy", "mode"),
        )

        # Downstream stages now operate on the repaired data.
        context.df = repairer.df
        context.metadata["repair_log"] = repairer.get_log()
        context.metadata["auto_repair_result"] = result.to_dict()

        issues = repairer.verify_clean(target_column=context.target_column)
        context.metadata["repair_verification"] = issues

        # Lineage: exactly what each column had done to it.
        context.record(
            self.name, result.summary(), event_type="auto_repair",
            detail=result.to_dict().get("columns_imputed", {}),
        )
        for column, info in result.columns_imputed.items():
            context.record(
                self.name,
                f"{info['strategy']} imputation with {info['fill_value']!r}, "
                f"{info['cells_filled']} cell(s) changed",
                event_type="imputed",
                column=column,
                detail=info,
            )
        if result.duplicates_removed:
            context.record(
                self.name, f"{result.duplicates_removed} duplicate row(s) removed",
                event_type="deduplicated",
                detail={"rows_removed": result.duplicates_removed},
            )
        if result.target_rows_dropped:
            context.record(
                self.name,
                f"{result.target_rows_dropped} row(s) dropped for a missing label",
                event_type="rows_dropped",
                column=context.target_column,
                detail={"rows_dropped": result.target_rows_dropped},
            )

        if context.option("export", True):
            csv_path, manifest_path = repairer.export(
                dataset_name=context.dataset_name or "dataset",
                target_column=context.target_column,
            )
            context.artifacts["repaired_csv"] = str(csv_path)
            context.artifacts["repair_manifest"] = str(manifest_path)

        return self._result(
            output={
                "columns_imputed": len(result.columns_imputed),
                "duplicates_removed": result.duplicates_removed,
                "rows_dropped_missing_target": result.target_rows_dropped,
                "missing_cells_remaining": int(repairer.df.isna().sum().sum()),
                "guarantees_met": not issues,
            },
        )


# ============================================================================
# Training
# ============================================================================


class TrainingStage(PipelineStage):
    """Train and compare candidate models, logging to MLflow."""

    def __init__(self) -> None:
        super().__init__("training")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("train", True):
            return "training disabled for this run"
        if not context.target_column:
            return "no target column identified, nothing to predict"
        df = context.df
        if df is not None and context.target_column not in df.columns:
            return f"target column '{context.target_column}' is not present in the data"
        if df is not None and len(df) < 20:
            return f"only {len(df)} rows available, too few to train and evaluate"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.ml.trainer import MLTrainingEngine

        df = context.require_df(self.name)
        engine = MLTrainingEngine(seed=context.option("seed", 42))
        model, metrics, save_path = engine.train(
            df,
            target_column=str(context.target_column),
            dataset_name=context.dataset_name,
        )

        context.metadata["training_metrics"] = metrics
        # Kept in memory (not serialised into the report) so the explainability
        # stage can work on the exact fitted pipeline, preprocessing included.
        context.metadata["trained_model"] = model
        # The trainer selects on f1 for classification and rmse for regression,
        # so the metric set identifies the task without a second inference.
        context.metadata["task_type"] = (
            "classification" if "f1_score" in metrics else "regression"
        )
        context.artifacts["model"] = str(save_path)

        # Lineage: which columns actually reached the model.
        context.record(
            self.name,
            f"Trained on {len(df):,} rows; best model saved to {Path(save_path).name}",
            event_type="trained",
            detail={"model_path": str(save_path)},
        )
        importances = metrics.get("feature_importances", {}) or {}
        for column in df.columns:
            if column == context.target_column:
                context.record(
                    self.name, "used as the prediction target",
                    event_type="target", column=str(column),
                )
                continue
            note = "included in training"
            if column in importances:
                note += f" (importance {importances[column]:.4f})"
            context.record(
                self.name, note, event_type="feature", column=str(column),
                detail={"importance": importances.get(column)},
            )

        headline = {
            k: round(float(v), 4)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "feature_importances"
        }
        return self._result(output={"model_path": str(save_path), **headline})


# ============================================================================
# Evaluation
# ============================================================================


class EvaluationStage(PipelineStage):
    """Measure how far model performance decays as the data is corrupted."""

    def __init__(self) -> None:
        super().__init__("evaluation")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("robustness", True):
            return "robustness evaluation disabled for this run"
        if not context.target_column:
            return "no target column identified"
        training = context.results.get("training")
        if training is None or not training.ok:
            return "training did not complete, so there is no baseline to degrade"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.robustness.evaluator import RobustnessEvaluator

        df = context.require_df(self.name)
        results = RobustnessEvaluator(seed=context.option("seed", 42)).evaluate_robustness(
            df,
            target_column=str(context.target_column),
            mutation_type=context.option("mutation_type", "missing_values"),
            levels=context.option("robustness_levels"),
        )
        context.metadata["robustness_results"] = results

        return self._result(output={
            "mutation_type": context.option("mutation_type", "missing_values"),
            "elasticity": results.get("elasticity_score"),
            "baseline": results.get("baseline_score"),
        })


# ============================================================================
# Drift
# ============================================================================


class DriftStage(PipelineStage):
    """Compare the dataset against a reference (training) dataset.

    Runs against the *ingested* data rather than the repaired data: the
    question is whether the incoming data resembles what the model was trained
    on, and imputing the gaps first would mask exactly the change worth seeing.
    """

    def __init__(self) -> None:
        super().__init__("drift")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("reference_path"):
            return "no reference dataset supplied (pass --reference to enable)"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.drift.detector import DriftDetector
        from src.ingestion.dataset_loader import DatasetLoader

        reference_path = Path(str(context.option("reference_path")))
        reference_df, _ = DatasetLoader().load(reference_path)

        # original_df is the ingested frame before repair.
        current_df = context.original_df if context.original_df is not None else context.require_df(self.name)

        report = DriftDetector().detect(
            reference_df,
            current_df,
            reference_name=reference_path.name,
            current_name=context.dataset_name or "current",
        )
        context.metadata["drift_report"] = report.to_dict()

        context.record(
            self.name, report.summary(), event_type="drift_checked",
            detail={"status": report.status, "overall_psi": report.overall_score},
        )
        for result in report.columns:
            if result.status != "STABLE":
                context.record(
                    self.name, f"{result.status.lower().replace('_', ' ')}: {result.explanation}",
                    event_type="drift", column=result.column,
                    detail={"status": result.status, "psi": result.psi},
                )

        return self._result(output={
            "status": report.status,
            "overall_psi": report.overall_score,
            "drifted_columns": len(report.drifted_columns),
            "compared_columns": len(report.compared_columns),
            "added_columns": len(report.added_columns),
            "removed_columns": len(report.removed_columns),
        })


# ============================================================================
# Explainability
# ============================================================================


class ExplainabilityStage(PipelineStage):
    """Explain the winning model: global importance and sample predictions."""

    def __init__(self) -> None:
        super().__init__("explainability")

    def should_skip(self, context: PipelineContext) -> Optional[str]:
        if not context.option("explain", True):
            return "explainability disabled for this run"
        training = context.results.get("training")
        if training is None or not training.ok:
            return "training did not complete, so there is no model to explain"
        return None

    def execute(self, context: PipelineContext) -> StageResult:
        from src.ml.explainer import ModelExplainer

        df = context.require_df(self.name)
        target = str(context.target_column)
        model = context.metadata.get("trained_model")
        if model is None:
            return self._result(
                StageStatus.FAILED,
                error="The training stage did not publish a fitted model to explain.",
            )

        frame = df.dropna(subset=[target])
        X = frame.drop(columns=[target])
        y = frame[target]

        task_type = context.metadata.get("task_type", "classification")
        explainer = ModelExplainer(model, task_type=task_type, seed=context.option("seed", 42))

        # Explain a small spread of rows rather than an arbitrary first few.
        sample_size = min(int(context.option("explain_rows", 3)), len(X))
        rows = list(np.linspace(0, len(X) - 1, sample_size).astype(int)) if sample_size else []

        report = explainer.explain(
            X, y,
            rows=rows,
            model_name=context.metadata.get("best_model_name", ""),
            target_column=target,
            n_repeats=context.option("permutation_repeats", 5),
        )
        context.metadata["explainability"] = report.to_dict()

        context.record(
            self.name, report.summary(), event_type="explained",
            detail={"method": report.method},
        )
        for entry in report.global_importance[:10]:
            context.record(
                self.name,
                f"importance rank {entry.rank} ({entry.importance:+.4f})",
                event_type="importance", column=entry.feature,
                detail={"rank": entry.rank, "importance": entry.importance},
            )

        top = [f.feature for f in report.global_importance[:3]]
        return self._result(output={
            "method": report.method,
            "top_features": ", ".join(top),
            "rows_explained": len(report.local_explanations),
        })


# ============================================================================
# Reporting
# ============================================================================


class ReportingStage(PipelineStage):
    """Compile every stage's findings into one consolidated report."""

    def __init__(self) -> None:
        super().__init__("reporting")

    def execute(self, context: PipelineContext) -> StageResult:
        from src.reporting.generator import ReportGenerator

        meta = context.metadata
        training_metrics = meta.get("training_metrics")

        # Capture the trace now that every earlier stage has recorded its work.
        if context.lineage is not None and "lineage" not in meta:
            try:
                meta["lineage"] = context.lineage.trace().to_dict()
            except Exception as exc:  # lineage must never fail the report
                self.logger.warning("Could not read lineage trace: %s", exc)

        report_path = ReportGenerator().export_report(
            dataset_name=context.dataset_name or "dataset",
            validation_report=meta.get("validation_report")
            or {"success": False, "score": 0.0, "ge_results": []},
            health_report=meta.get("health_report_after")
            or meta.get("health_report")
            or {"overall_score": 0.0, "grade": "F", "dimensions": {}, "explanations": []},
            audit_report=meta.get("quality_report") or {"findings": [], "summary": {}},
            repair_log=meta.get("repair_log") or [],
            robustness_results=meta.get("robustness_results"),
            training_metrics=training_metrics,
            format_type=context.option("report_format", "html"),
            drift_report=meta.get("drift_report"),
            version_comparison=meta.get("version_comparison"),
            lineage=meta.get("lineage"),
            explainability=meta.get("explainability"),
        )
        context.artifacts["report"] = str(report_path)

        return self._result(output={"report_path": str(report_path)})


__all__ = [
    "DriftStage",
    "ExplainabilityStage",
    "EvaluationStage",
    "HealthStage",
    "IngestionStage",
    "ProfilingStage",
    "QualityStage",
    "RepairStage",
    "ReportingStage",
    "TrainingStage",
    "ValidationStage",
]
