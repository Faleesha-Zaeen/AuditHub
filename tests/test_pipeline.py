"""
Tests for src.pipeline
=======================

The orchestrator's central promise: a run that did not do the work must not
report success. The previous skeleton caught ``NotImplementedError`` from every
stage and still returned ``COMPLETED``.
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    Pipeline,
    PipelineContext,
    PipelineStatus,
    StageStatus,
    create_default_pipeline,
)
from src.pipeline.core import PipelineStage, StageResult
from src.pipeline.stages import (
    HealthStage,
    IngestionStage,
    QualityStage,
    RepairStage,
)


@pytest.fixture
def dataset_path(tmp_path):
    """A small but genuinely messy dataset written to disk."""
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "age": rng.integers(20, 70, n).astype(float),
        "amount": rng.normal(500, 120, n).round(0),
        "city": rng.choice(["Delhi", "Pune", "Mumbai"], n),
        "label": rng.choice(["yes", "no"], n),
    })
    df.loc[rng.choice(n, 8, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "city"] = "N/A"
    df["amount"] = df["amount"].apply(lambda v: f"${v:,.0f}")
    df["blank"] = np.nan
    path = tmp_path / "pipe.csv"
    df.to_csv(path, index=False)
    return path


def _fast_pipeline() -> Pipeline:
    """Ingestion, quality, health and repair only -- no profiling or training."""
    return Pipeline(
        name="fast",
        stages=[IngestionStage(), QualityStage(), HealthStage("health"),
                RepairStage(), HealthStage("health_after")],
    )


class TestPipelineExecution:
    """A real run over a real file."""

    def test_all_stages_complete(self, dataset_path, tmp_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path,
            dataset_name="pipe.csv",
            target_column="label",
            config={"export": True, "register": False},
        ))
        assert pipeline.status is PipelineStatus.COMPLETED
        for name in ("ingestion", "quality", "health", "repair", "health_after"):
            assert ctx.results[name].status is StageStatus.COMPLETED, name

    def test_repair_stage_actually_repairs(self, dataset_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path, target_column="label",
            config={"export": False, "register": False},
        ))
        assert int(ctx.df.isna().sum().sum()) == 0
        assert ctx.results["repair"].output["guarantees_met"] is True

    def test_downstream_stages_see_repaired_data(self, dataset_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path, target_column="label",
            config={"export": False, "register": False},
        ))
        before = ctx.results["health"].output["score"]
        after = ctx.results["health_after"].output["score"]
        assert after >= before, "health must not get worse after repair"
        assert ctx.original_df is not None
        assert int(ctx.original_df.isna().sum().sum()) > 0

    def test_stage_durations_are_recorded(self, dataset_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path, config={"export": False, "register": False}
        ))
        assert all(
            r.duration_seconds is not None
            for r in ctx.results.values()
            if r.status is not StageStatus.SKIPPED
        )


class TestFailureHandling:
    """Failures must be visible, not swallowed."""

    def test_missing_file_fails_the_run(self, tmp_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(dataset_path=tmp_path / "nope.csv"))
        assert pipeline.status is PipelineStatus.FAILED
        assert ctx.results["ingestion"].status is StageStatus.FAILED
        assert ctx.results["ingestion"].error

    def test_critical_failure_aborts_remaining_stages(self, tmp_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(dataset_path=tmp_path / "nope.csv"))
        # Nothing after the critical ingestion stage should have run.
        assert "quality" not in ctx.results

    def test_a_raising_stage_is_recorded_as_failed(self, dataset_path):
        class Exploding(PipelineStage):
            def __init__(self):
                super().__init__("exploding")

            def execute(self, context):
                raise RuntimeError("boom")

        pipeline = Pipeline(name="t", stages=[IngestionStage(), Exploding()])
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path, config={"register": False}
        ))
        assert ctx.results["exploding"].status is StageStatus.FAILED
        assert "boom" in ctx.results["exploding"].error
        # Non-critical failure with other stages succeeding -> partial.
        assert pipeline.status is PipelineStatus.PARTIALLY_COMPLETED

    def test_empty_pipeline_completes(self):
        pipeline = Pipeline(name="empty", stages=[])
        pipeline.run(PipelineContext())
        assert pipeline.status is PipelineStatus.COMPLETED


class TestStageSkipping:
    """Stages that cannot run should skip with a stated reason."""

    def test_training_skips_without_a_target(self, dataset_path):
        from src.pipeline.stages import TrainingStage

        pipeline = Pipeline(name="t", stages=[IngestionStage(), TrainingStage()])
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path,
            config={"register": False, "train": True},
        ))
        result = ctx.results["training"]
        if result.status is StageStatus.SKIPPED:
            assert "reason" in result.output
        else:
            # A target was inferred from the data, which is also valid.
            assert ctx.target_column

    def test_disabled_stage_is_skipped(self, dataset_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path,
            config={"repair": False, "register": False},
        ))
        assert ctx.results["repair"].status is StageStatus.SKIPPED
        assert pipeline.status is PipelineStatus.COMPLETED


class TestDefaultPipeline:
    """Shape of the shipped pipeline."""

    def test_contains_every_stage_in_order(self):
        pipeline = create_default_pipeline()
        names = [s.name for s in pipeline.stages]
        assert names == [
            "ingestion", "validation", "drift", "profiling", "quality", "health",
            "repair", "health_after", "training", "evaluation", "explainability",
            "reporting",
        ]

    def test_drift_skips_without_a_reference(self, dataset_path):
        # The drift stage must not disturb the existing flow when no reference
        # dataset is supplied.
        pipeline = create_default_pipeline()
        stage = next(s for s in pipeline.stages if s.name == "drift")
        assert stage.should_skip(PipelineContext(dataset_path=dataset_path))

    def test_ingestion_is_the_only_critical_stage(self):
        pipeline = create_default_pipeline()
        critical = [s.name for s in pipeline.stages if s.critical]
        assert critical == ["ingestion"]

    def test_starts_not_started(self):
        assert create_default_pipeline().status is PipelineStatus.NOT_STARTED

    def test_summary_rows_cover_all_stages(self, dataset_path):
        pipeline = _fast_pipeline()
        ctx = pipeline.run(PipelineContext(
            dataset_path=dataset_path, config={"export": False, "register": False}
        ))
        rows = pipeline.summary_rows(ctx)
        assert len(rows) == len(pipeline.stages)
        assert all("status" in r for r in rows)


class TestCli:
    """The command-line entry point."""

    def test_missing_file_exits_with_usage_code(self, tmp_path):
        from src.pipeline.__main__ import main

        code = main(["run", str(tmp_path / "missing.csv")])
        assert code == 3

    def test_successful_run_exits_zero(self, dataset_path, capsys):
        from src.pipeline.__main__ import main

        code = main([
            "run", str(dataset_path), "--target", "label",
            "--no-profile", "--no-train", "--no-robustness", "--json",
        ])
        captured = capsys.readouterr().out
        assert code == 0, captured
        assert '"status": "COMPLETED"' in captured
