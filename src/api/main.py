"""
AuditHub API - FastAPI Backend Entrypoint
===========================================

Exposes REST APIs for every major AuditHub engine module.
"""

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import joblib

from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_analyzer import DatasetAnalyzer
from src.validation.validator import DatasetValidator
from src.profiling.profiler import DatasetProfiler
from src.quality.auditor import DatasetAuditor
from src.health.calculator import HealthScoreCalculator
from src.drift.detector import DriftDetector
from src.lineage.comparator import VersionComparator
from src.lineage.tracker import LineageTracker
from src.repair.repairer import DatasetRepairer
from src.utils.dataset_registry import DatasetRegistry
from src.mutation_lab.mutator import DatasetMutator
from src.robustness.evaluator import RobustnessEvaluator
from src.ml.explainer import ModelExplainer
from src.ml.trainer import MLTrainingEngine
from src.reporting.generator import ReportGenerator
from src.utils.constants import RAW_DATA_DIR, DATABASE_PATH
from src.utils.logger import get_logger
from src.utils.exceptions import AuditHubException

logger = get_logger(__name__)


def serialize_for_api(obj: Any) -> Any:
    """Recursively convert numpy types and dataclasses to native JSON-serializable types."""
    import numpy as np
    from dataclasses import is_dataclass, asdict

    if isinstance(obj, dict):
        return {str(k): serialize_for_api(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set, np.ndarray)):
        return [serialize_for_api(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif is_dataclass(obj):
        return serialize_for_api(asdict(obj))
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return serialize_for_api(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        return serialize_for_api(vars(obj))
    else:
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        return str(obj)


app = FastAPI(
    title="AuditHub API",
    description="REST API for AuditHub Dataset Quality & MLOps Platform",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(AuditHubException)
async def audithub_exception_handler(request, exc):
    logger.error("AuditHub error caught by API: %s", exc.message)
    return {
        "status": "error",
        "error_type": exc.__class__.__name__,
        "message": exc.message,
        "details": exc.details
    }


# Helper: Load uploaded file to DataFrame
def _load_uploaded_file(file: UploadFile) -> Tuple[pd.DataFrame, Path]:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = RAW_DATA_DIR / file.filename
    with open(temp_path, "wb") as f:
        f.write(file.file.read())
    
    loader = DatasetLoader()
    df, _ = loader.load(temp_path)
    return df, temp_path


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "app": "AuditHub API",
        "status": "healthy",
        "database": str(DATABASE_PATH.resolve()),
        "endpoints": {
            "ingest": "POST /api/v1/ingest",
            "validate": "POST /api/v1/validate",
            "profile": "POST /api/v1/profile",
            "audit": "POST /api/v1/audit",
            "health": "POST /api/v1/health",
            "repair": "POST /api/v1/repair",
            "drift": "POST /api/v1/drift",
            "compare_versions": "POST /api/v1/versions/compare",
            "list_version_groups": "GET /api/v1/versions",
            "list_versions": "GET /api/v1/versions/{version_group}",
            "list_lineage_runs": "GET /api/v1/lineage",
            "lineage": "GET /api/v1/lineage/{identifier}",
            "explain": "POST /api/v1/explain",
            "mutate": "POST /api/v1/mutate",
            "robustness": "POST /api/v1/robustness",
            "train": "POST /api/v1/train",
        }
    }


@app.post("/api/v1/ingest")
async def ingest_dataset(file: UploadFile = File(...)):
    """Upload and analyze a dataset, returning its summary statistics."""
    try:
        df, temp_path = _load_uploaded_file(file)
        analyzer = DatasetAnalyzer()
        summary = analyzer.analyze(df, source_filename=file.filename)
        return serialize_for_api({
            "status": "success",
            "file_path": str(temp_path.resolve()),
            "summary": summary.to_dict()
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/validate")
async def validate_dataset(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    """Run schema and Great Expectations validations on the dataset."""
    try:
        df, _ = _load_uploaded_file(file)
        validator = DatasetValidator()
        report = validator.validate(df, target_column=target_column, dataset_name=file.filename)
        return serialize_for_api({
            "status": "success",
            "report": report.to_dict()
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/profile")
async def profile_dataset(file: UploadFile = File(...)):
    """Trigger ydata-profiling and return summary statistics."""
    try:
        df, _ = _load_uploaded_file(file)
        profiler = DatasetProfiler()
        profiler.minimal = True  # force minimal mode for API performance
        summary = profiler.profile(df, dataset_name=file.filename)
        return serialize_for_api({
            "status": "success",
            "summary": summary
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/audit")
async def audit_dataset(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    """Run rule-based dataset quality audit checks."""
    try:
        df, _ = _load_uploaded_file(file)
        auditor = DatasetAuditor()
        report = auditor.audit(df, target_column=target_column)
        return serialize_for_api({
            "status": "success",
            "report": report.to_dict()
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/health")
async def health_score_dataset(file: UploadFile = File(...), target_column: Optional[str] = Form(None)):
    """Calculate multi-dimensional dataset health scores and grade."""
    try:
        df, _ = _load_uploaded_file(file)
        calculator = HealthScoreCalculator()
        report = calculator.calculate(df, target_column=target_column)
        return serialize_for_api({
            "status": "success",
            "report": report.to_dict()
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/repair")
async def repair_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    numeric_strategy: str = Form("median"),
    categorical_strategy: str = Form("mode"),
    download: bool = Form(False),
):
    """Automatically repair a dataset and return the repaired data.

    Set ``download=true`` to receive the repaired dataset as a CSV file
    instead of a JSON payload.
    """
    try:
        df, _ = _load_uploaded_file(file)
        repairer = DatasetRepairer(df)

        result = repairer.auto_repair(
            target_column=target_column,
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )

        if download:
            filename = Path(file.filename or "dataset").stem + "_repaired.csv"
            return StreamingResponse(
                io.BytesIO(repairer.to_csv_bytes()),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        repaired = repairer.df
        return serialize_for_api({
            "status": "success",
            "shape": {"rows": int(len(repaired)), "columns": int(len(repaired.columns))},
            "repaired_columns": [str(c) for c in repaired.columns],
            "dtypes": {str(c): str(t) for c, t in repaired.dtypes.items()},
            "summary": result.to_dict(),
            "verification": {
                "issues": repairer.verify_clean(target_column=target_column),
                "passed": not repairer.verify_clean(target_column=target_column),
            },
            "repair_log": repairer.get_log(),
            # The repaired data itself, as records -- previously the caller got
            # only column names and a log, with no way to retrieve the result.
            "data": json.loads(repaired.to_json(orient="records", date_format="iso")),
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/drift")
async def detect_drift(
    reference_file: UploadFile = File(..., description="Reference/training dataset"),
    current_file: UploadFile = File(..., description="Current/new dataset"),
    columns: Optional[str] = Form(None),
    warning_threshold: Optional[float] = Form(None),
    critical_threshold: Optional[float] = Form(None),
):
    """Compare a current dataset against a reference dataset.

    Answers whether new data still resembles the data a model was trained on.
    Returns a per-column PSI, a significance test, a Stable/Warning/Critical
    verdict per column, and an overall dataset drift score.
    """
    try:
        reference_df, _ = _load_uploaded_file(reference_file)
        current_df, _ = _load_uploaded_file(current_file)

        overrides: Dict[str, Any] = {}
        if warning_threshold is not None:
            overrides["psi_warning_threshold"] = warning_threshold
        if critical_threshold is not None:
            overrides["psi_critical_threshold"] = critical_threshold

        selected = [c.strip() for c in columns.split(",")] if columns else None
        report = DriftDetector(thresholds=overrides or None).detect(
            reference_df,
            current_df,
            columns=selected,
            reference_name=reference_file.filename or "reference",
            current_name=current_file.filename or "current",
        )
        return serialize_for_api({"status": "success", "drift": report.to_dict()})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/versions/compare")
async def compare_versions(
    left_file: UploadFile = File(..., description="Older version (V1)"),
    right_file: UploadFile = File(..., description="Newer version (V2)"),
    target_column: Optional[str] = Form(None),
    detect_drift: bool = Form(True),
):
    """Compare two versions of a dataset.

    Reports row/column deltas, added, removed and renamed columns, type
    changes, missing-value and duplicate changes, target changes, and
    distribution shifts.
    """
    try:
        left_df, _ = _load_uploaded_file(left_file)
        right_df, _ = _load_uploaded_file(right_file)

        comparison = VersionComparator(detect_drift=detect_drift).compare(
            left_df,
            right_df,
            left_name=left_file.filename or "V1",
            right_name=right_file.filename or "V2",
            target_column=target_column,
        )
        return serialize_for_api({"status": "success", "comparison": comparison.to_dict()})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/versions")
def list_version_groups():
    """List logical datasets in the registry with their version counts."""
    try:
        return serialize_for_api({
            "status": "success",
            "version_groups": DatasetRegistry().list_version_groups(),
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/versions/{version_group}")
def list_versions(version_group: str):
    """List every registered version of one logical dataset."""
    try:
        versions = DatasetRegistry().list_versions(version_group)
        if not versions:
            raise HTTPException(
                status_code=404, detail=f"No versions registered for '{version_group}'."
            )
        return serialize_for_api({
            "status": "success",
            "version_group": version_group,
            "versions": [v.to_dict() for v in versions],
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/lineage")
def list_lineage_runs(limit: int = 50):
    """List recorded pipeline runs."""
    try:
        return serialize_for_api({
            "status": "success",
            "runs": LineageTracker().list_runs(limit=limit),
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/v1/lineage/{identifier}")
def get_lineage(identifier: str, column: Optional[str] = None):
    """Return the lineage trace for a run ID or dataset ID.

    Pass ``column`` to get that single column's journey through the pipeline.
    """
    try:
        tracker = LineageTracker()
        trace = tracker.trace(identifier)
        if not trace.events:
            trace = tracker.trace_for_dataset(identifier)
        if not trace.events:
            raise HTTPException(
                status_code=404, detail=f"No lineage recorded for '{identifier}'."
            )

        if column:
            return serialize_for_api({
                "status": "success",
                "run_id": trace.run_id,
                "column": column,
                "history": [e.to_dict() for e in trace.column_history(column)],
            })
        return serialize_for_api({"status": "success", "lineage": trace.to_dict()})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/explain")
async def explain_model(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    rows: Optional[str] = Form(None),
    top_n: int = Form(10),
    permutation_repeats: int = Form(5),
):
    """Train a model on the uploaded dataset and explain its predictions.

    Returns global feature importance plus, for each requested row, the
    features pushing the prediction toward and against its outcome.
    ``rows`` is a comma-separated list of row positions.
    """
    try:
        df, _ = _load_uploaded_file(file)
        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' is not in the dataset.",
            )

        model, metrics, model_path = MLTrainingEngine().train(
            df, target_column=target_column, dataset_name=file.filename
        )
        task_type = "classification" if "f1_score" in metrics else "regression"

        frame = df.dropna(subset=[target_column])
        X = frame.drop(columns=[target_column])
        y = frame[target_column]

        selected = [int(r.strip()) for r in rows.split(",")] if rows else [0]
        report = ModelExplainer(model, task_type=task_type).explain(
            X, y, rows=selected, target_column=target_column,
            n_repeats=permutation_repeats, top_n=top_n,
        )

        return serialize_for_api({
            "status": "success",
            "model_path": str(model_path),
            "metrics": {k: v for k, v in metrics.items() if k != "feature_importances"},
            "explainability": report.to_dict(),
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/mutate")
async def mutate_dataset(
    file: UploadFile = File(...),
    mutation_type: str = Form(...),
    fraction: float = Form(...)
):
    """Mutate dataset by injecting missing values, noise, or outliers."""
    try:
        df, _ = _load_uploaded_file(file)
        mutator = DatasetMutator()
        
        if mutation_type == "missing_values":
            df_mut = mutator.inject_missing_values(df, fraction=fraction)
        elif mutation_type == "gaussian_noise":
            df_mut = mutator.inject_gaussian_noise(df, noise_level=fraction)
        elif mutation_type == "outliers":
            df_mut = mutator.inject_outliers(df, fraction=fraction)
        elif mutation_type == "duplicates":
            df_mut = mutator.inject_duplicates(df, fraction=fraction)
        else:
            raise ValueError(f"Unknown mutation type: {mutation_type}")
            
        return serialize_for_api({
            "status": "success",
            "mutation_applied": mutation_type,
            "shape": df_mut.shape,
            "columns": list(df_mut.columns)
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/robustness")
async def evaluate_robustness(
    file: UploadFile = File(...),
    target_column: str = Form(...),
    mutation_type: str = Form(...)
):
    """Run model robustness decay evaluations."""
    try:
        df, _ = _load_uploaded_file(file)
        evaluator = RobustnessEvaluator()
        results = evaluator.evaluate_robustness(df, target_column=target_column, mutation_type=mutation_type)
        return serialize_for_api({
            "status": "success",
            "results": results
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/v1/train")
async def train_model(file: UploadFile = File(...), target_column: str = Form(...)):
    """Auto-train ML model, register it to MLflow, and return best model metrics."""
    try:
        df, _ = _load_uploaded_file(file)
        trainer = MLTrainingEngine()
        model, metrics, save_path = trainer.train(df, target_column=target_column, dataset_name=file.filename)
        return serialize_for_api({
            "status": "success",
            "model_path": save_path,
            "metrics": {k: v for k, v in metrics.items() if k != "feature_importances"}
        })
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
