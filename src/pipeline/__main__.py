"""
AuditHub - Pipeline CLI
========================

Headless entry point for the full audit pipeline::

    python -m src.pipeline run data/raw/sales.csv --target revenue

Exits non-zero when the run fails, so it can gate a CI job or a cron task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from src.pipeline import PipelineStatus, run_pipeline

# Exit codes: 0 success, 1 partial (some stage failed), 2 failed, 3 bad usage.
_EXIT_CODES = {
    PipelineStatus.COMPLETED: 0,
    PipelineStatus.PARTIALLY_COMPLETED: 1,
    PipelineStatus.FAILED: 2,
}


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the AuditHub quality pipeline over a dataset.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the full pipeline over a dataset file.")
    run.add_argument("dataset", type=Path, help="Path to the dataset file.")
    run.add_argument("--target", default=None, help="Target/label column name.")
    run.add_argument(
        "--numeric-strategy", default="median", choices=["median", "mean", "mode"],
        help="Imputation strategy for numeric columns (default: median).",
    )
    run.add_argument(
        "--categorical-strategy", default="mode", choices=["mode", "constant"],
        help="Imputation strategy for categorical columns (default: mode).",
    )
    run.add_argument("--no-profile", action="store_true", help="Skip ydata-profiling.")
    run.add_argument("--no-repair", action="store_true", help="Skip the repair stage.")
    run.add_argument("--no-train", action="store_true", help="Skip model training.")
    run.add_argument("--no-robustness", action="store_true", help="Skip robustness evaluation.")
    run.add_argument("--no-export", action="store_true", help="Do not write the repaired CSV.")
    run.add_argument(
        "--report-format", default="html", choices=["html", "json"],
        help="Consolidated report format (default: html).",
    )
    run.add_argument("--max-rows", type=int, default=None, help="Read at most N rows.")
    run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    run.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    run.add_argument(
        "--reference", type=Path, default=None,
        help="Reference/training dataset. Enables the drift stage.",
    )

    drift = sub.add_parser(
        "drift", help="Compare a current dataset against a reference dataset."
    )
    drift.add_argument("reference", type=Path, help="Reference (training) dataset.")
    drift.add_argument("current", type=Path, help="Current (new) dataset.")
    drift.add_argument(
        "--columns", default=None,
        help="Comma-separated subset of columns to compare.",
    )
    drift.add_argument(
        "--warning-threshold", type=float, default=None,
        help="PSI at which a column is flagged WARNING (default 0.10).",
    )
    drift.add_argument(
        "--critical-threshold", type=float, default=None,
        help="PSI at which a column is flagged CRITICAL (default 0.25).",
    )
    drift.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    drift.add_argument(
        "--fail-on", default="none", choices=["none", "warning", "critical"],
        help="Exit non-zero when drift reaches this level (default: none).",
    )

    compare = sub.add_parser(
        "compare", help="Compare two versions of a dataset."
    )
    compare.add_argument("left", type=Path, help="Older version (V1).")
    compare.add_argument("right", type=Path, help="Newer version (V2).")
    compare.add_argument("--target", default=None, help="Target/label column name.")
    compare.add_argument(
        "--no-drift", action="store_true",
        help="Schema-only comparison; skip the distribution check.",
    )
    compare.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")

    lineage = sub.add_parser(
        "lineage", help="Show what the pipeline did to a dataset."
    )
    lineage.add_argument(
        "identifier", nargs="?", default=None,
        help="Run ID or dataset ID. Omit to list recent runs.",
    )
    lineage.add_argument(
        "--column", default=None,
        help="Show one column's history through the pipeline.",
    )
    lineage.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")

    explain = sub.add_parser(
        "explain", help="Train a model on a dataset and explain its predictions."
    )
    explain.add_argument("dataset", type=Path, help="Dataset file.")
    explain.add_argument("--target", required=True, help="Target/label column name.")
    explain.add_argument(
        "--row", type=int, action="append", default=None,
        help="Row to explain individually. Repeatable.",
    )
    explain.add_argument(
        "--top", type=int, default=10, help="Features to show (default: 10).",
    )
    explain.add_argument(
        "--repeats", type=int, default=5,
        help="Permutation importance repeats (default: 5).",
    )
    explain.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")

    return parser


def _print_table(rows: List[dict]) -> None:
    """Print the per-stage summary as an aligned table."""
    headers = ("STAGE", "STATUS", "SECONDS", "DETAIL")
    widths = [
        max(len(headers[0]), *(len(str(r["stage"])) for r in rows)),
        max(len(headers[1]), *(len(str(r["status"])) for r in rows)),
        7,
    ]
    print(f"{headers[0]:<{widths[0]}}  {headers[1]:<{widths[1]}}  {headers[2]:>{widths[2]}}  {headers[3]}")
    print("-" * (sum(widths) + 6 + 40))
    for r in rows:
        seconds = f"{r['seconds']:.2f}" if r["seconds"] is not None else "-"
        detail = str(r["detail"] or "")
        if len(detail) > 60:
            detail = detail[:57] + "..."
        print(f"{r['stage']:<{widths[0]}}  {r['status']:<{widths[1]}}  {seconds:>{widths[2]}}  {detail}")


def _run_drift(args) -> int:
    """Handle the ``drift`` subcommand.

    Returns
    -------
    int
        ``3`` for bad usage, ``4`` when ``--fail-on`` is triggered, else ``0``.
    """
    from src.drift.detector import DriftDetector, DriftStatus
    from src.ingestion.dataset_loader import DatasetLoader

    for path, label in ((args.reference, "reference"), (args.current, "current")):
        if not path.exists():
            print(f"error: {label} dataset not found: {path}", file=sys.stderr)
            return 3

    loader = DatasetLoader()
    reference_df, _ = loader.load(args.reference)
    current_df, _ = loader.load(args.current)

    overrides = {}
    if args.warning_threshold is not None:
        overrides["psi_warning_threshold"] = args.warning_threshold
    if args.critical_threshold is not None:
        overrides["psi_critical_threshold"] = args.critical_threshold

    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    report = DriftDetector(thresholds=overrides or None).detect(
        reference_df,
        current_df,
        columns=columns,
        reference_name=args.reference.name,
        current_name=args.current.name,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print()
        print(f"AuditHub drift - {args.reference.name} -> {args.current.name}")
        print(f"rows: {report.reference_rows:,} -> {report.current_rows:,}")
        print()
        header = f"{'COLUMN':<24} {'STATUS':<16} {'KIND':<12} {'PSI':>9} {'P-VALUE':>10}"
        print(header)
        print("-" * len(header))
        for col in report.columns:
            psi = f"{col.psi:.4f}" if col.psi is not None else "-"
            p_value = f"{col.p_value:.3g}" if col.p_value is not None else "-"
            print(
                f"{col.column[:24]:<24} {col.status:<16} {col.column_kind:<12} "
                f"{psi:>9} {p_value:>10}"
            )
        print()
        for col in report.columns:
            if col.status in DriftStatus.ACTIONABLE:
                print(f"  * {col.explanation}")
        print()
        print(f"overall: {report.summary()}")

    if args.fail_on == "critical" and report.status == DriftStatus.CRITICAL:
        return 4
    if args.fail_on == "warning" and report.status in (
        DriftStatus.WARNING, DriftStatus.CRITICAL
    ):
        return 4
    return 0


def _run_compare(args) -> int:
    """Handle the ``compare`` subcommand."""
    from src.ingestion.dataset_loader import DatasetLoader
    from src.lineage.comparator import VersionComparator

    for path, label in ((args.left, "left"), (args.right, "right")):
        if not path.exists():
            print(f"error: {label} dataset not found: {path}", file=sys.stderr)
            return 3

    loader = DatasetLoader()
    left, _ = loader.load(args.left)
    right, _ = loader.load(args.right)

    comparison = VersionComparator(detect_drift=not args.no_drift).compare(
        left, right,
        left_name=args.left.name, right_name=args.right.name,
        target_column=args.target,
    )

    if args.json:
        print(json.dumps(comparison.to_dict(), indent=2, default=str))
        return 0

    print()
    print(f"AuditHub version comparison - {args.left.name} -> {args.right.name}")
    print()
    print(f"  Rows:       {comparison.left_rows:,} -> {comparison.right_rows:,} "
          f"({comparison.row_delta:+,})")
    print(f"  Columns:    {comparison.left_columns} -> {comparison.right_columns} "
          f"({comparison.column_delta:+d})")
    print(f"  Missing:    {comparison.left_missing_pct:.2f}% -> "
          f"{comparison.right_missing_pct:.2f}%")
    print(f"  Duplicates: {comparison.left_duplicates} -> {comparison.right_duplicates}")

    if comparison.added_columns:
        print("\n  Added:")
        for col in comparison.added_columns:
            print(f"    + {col}")
    if comparison.removed_columns:
        print("\n  Removed:")
        for col in comparison.removed_columns:
            print(f"    - {col}")
    if comparison.renamed_columns:
        print("\n  Renamed:")
        for rename in comparison.renamed_columns:
            print(f"    {rename['from']} -> {rename['to']}  ({rename['evidence']})")

    for severity in ("CRITICAL", "WARNING", "INFO"):
        changes = comparison.changes_by_severity(severity)
        if not changes:
            continue
        print(f"\n  {severity}:")
        for change in changes:
            print(f"    * {change.detail}")

    print(f"\nsummary: {comparison.summary()}")
    return 0


def _run_lineage(args) -> int:
    """Handle the ``lineage`` subcommand."""
    from src.lineage.tracker import LineageTracker

    tracker = LineageTracker()

    if not args.identifier:
        runs = tracker.list_runs()
        if args.json:
            print(json.dumps(runs, indent=2, default=str))
            return 0
        if not runs:
            print("No pipeline runs have been recorded yet.")
            return 0
        print()
        print(f"{'RUN ID':<38} {'DATASET':<24} {'EVENTS':>7}  STAGES")
        print("-" * 100)
        for run in runs:
            print(f"{run['run_id']:<38} {str(run['dataset_name'])[:24]:<24} "
                  f"{run['events']:>7}  {', '.join(run['stages'])}")
        print("\nPass a run id to see its lineage.")
        return 0

    trace = tracker.trace(args.identifier)
    if not trace.events:
        trace = tracker.trace_for_dataset(args.identifier)
    if not trace.events:
        print(f"error: no lineage recorded for '{args.identifier}'", file=sys.stderr)
        return 3

    if args.column:
        history = trace.column_history(args.column)
        if args.json:
            print(json.dumps([e.to_dict() for e in history], indent=2, default=str))
            return 0
        if not history:
            print(f"No lineage recorded for column '{args.column}'.")
            return 0
        print()
        print(args.column)
        for event in history:
            print(f"  -> {event.summary}  [{event.stage}]")
        return 0

    if args.json:
        print(json.dumps(trace.to_dict(), indent=2, default=str))
        return 0

    print()
    print(f"AuditHub lineage - {trace.dataset_name or trace.dataset_id or trace.run_id}")
    print(f"run: {trace.run_id}")
    print()
    print(f"  {trace.flow()}")
    print()
    print(f"{'STAGE':<16} {'EVENTS':>7}  SUMMARY")
    print("-" * 90)
    for stage in trace.stages():
        print(f"{stage['stage']:<16} {stage['event_count']:>7}  {stage['summary'][:60]}")

    columns = trace.columns()
    if columns:
        print("\ncolumn histories:")
        for name, events in list(columns.items())[:15]:
            print(f"\n  {name}")
            for event in events:
                print(f"    -> {event['summary']}  [{event['stage']}]")
    return 0


def _run_explain(args) -> int:
    """Handle the ``explain`` subcommand."""
    from src.ingestion.dataset_loader import DatasetLoader
    from src.ml.explainer import ModelExplainer
    from src.ml.trainer import MLTrainingEngine

    if not args.dataset.exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 3

    df, _ = DatasetLoader().load(args.dataset)
    if args.target not in df.columns:
        print(f"error: target column '{args.target}' not in the dataset", file=sys.stderr)
        return 3

    model, metrics, model_path = MLTrainingEngine().train(
        df, target_column=args.target, dataset_name=args.dataset.name
    )
    task_type = "classification" if "f1_score" in metrics else "regression"

    frame = df.dropna(subset=[args.target])
    X = frame.drop(columns=[args.target])
    y = frame[args.target]

    rows = args.row if args.row else [0]
    report = ModelExplainer(model, task_type=task_type).explain(
        X, y, rows=rows, target_column=args.target,
        n_repeats=args.repeats, top_n=args.top,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
        return 0

    print()
    print(f"AuditHub explainability - {args.dataset.name}")
    print(f"model: {report.model_name} ({task_type}) | method: {report.method}")
    print()
    print("GLOBAL FEATURE IMPORTANCE")
    print(f"{'#':>3}  {'FEATURE':<24} {'IMPORTANCE':>12}  {'':<22}")
    print("-" * 66)
    top = report.top_features(args.top)
    peak = max((abs(f.importance) for f in top), default=1.0) or 1.0
    for entry in top:
        bar = "#" * int(abs(entry.importance) / peak * 20)
        print(f"{entry.rank:>3}  {entry.feature[:24]:<24} {entry.importance:>12.6f}  {bar:<22}")

    for local in report.local_explanations:
        print()
        print(f"ROW {local.row_index} - predicted: {local.prediction}", end="")
        if local.confidence is not None:
            print(f" (confidence {local.confidence:.1%})")
        else:
            print()
        toward = local.pushing_toward()
        against = local.pushing_against()
        if toward:
            print("  pushing toward this outcome:")
            for rank, c in enumerate(toward[:5], start=1):
                print(f"    {rank}. {c.feature:<20} = {str(c.value)[:18]:<18} {c.contribution:+.6f}")
        if against:
            print("  pushing against:")
            for rank, c in enumerate(against[:5], start=1):
                print(f"    {rank}. {c.feature:<20} = {str(c.value)[:18]:<18} {c.contribution:+.6f}")

    print()
    print(f"model saved to: {model_path}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI.

    Returns
    -------
    int
        Process exit code.
    """
    args = _build_parser().parse_args(argv)

    if args.command == "drift":
        return _run_drift(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "lineage":
        return _run_lineage(args)
    if args.command == "explain":
        return _run_explain(args)

    if not args.dataset.exists():
        print(f"error: dataset not found: {args.dataset}", file=sys.stderr)
        return 3
    if args.reference is not None and not args.reference.exists():
        print(f"error: reference dataset not found: {args.reference}", file=sys.stderr)
        return 3

    pipeline, context = run_pipeline(
        args.dataset,
        target_column=args.target,
        reference_path=str(args.reference) if args.reference else None,
        numeric_strategy=args.numeric_strategy,
        categorical_strategy=args.categorical_strategy,
        profile=not args.no_profile,
        repair=not args.no_repair,
        train=not args.no_train,
        robustness=not args.no_robustness,
        export=not args.no_export,
        report_format=args.report_format,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    rows = pipeline.summary_rows(context)
    exit_code = _EXIT_CODES.get(pipeline.status, 2)

    if args.json:
        print(json.dumps({
            "status": pipeline.status.name,
            "dataset": str(args.dataset),
            "target_column": context.target_column,
            "stages": rows,
            "artifacts": context.artifacts,
        }, indent=2, default=str))
        return exit_code

    print()
    print(f"AuditHub pipeline - {args.dataset.name}")
    print(f"target column: {context.target_column or '(none identified)'}")
    print()
    _print_table(rows)
    print()

    before = context.metadata.get("health_report")
    after = context.metadata.get("health_report_after")
    if before and after:
        print(
            f"health: {before.get('grade')} ({before.get('overall_score')})"
            f"  ->  {after.get('grade')} ({after.get('overall_score')})"
        )

    if context.artifacts:
        print("\nartifacts:")
        for key, path in context.artifacts.items():
            print(f"  {key:18} {path}")

    print(f"\nstatus: {pipeline.status.name}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
