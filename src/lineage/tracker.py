"""
AuditHub Lineage - Pipeline Lineage Tracking
=============================================

Records what each pipeline stage did, to the dataset as a whole and to
individual columns, so a run can be replayed as a story::

    Raw Dataset -> Ingestion -> Validation -> Profiling -> Audit
                -> Repair -> Re-audit -> Training -> Model

and a single column can be followed through it::

    age
      -> 143 missing values detected      (ingestion)
      -> median imputation, 143 cells changed  (repair)
      -> included in training             (training)

Events live in the existing ``data/audithub.db`` alongside the dataset
registry -- a second store would have to be kept in sync with it for no gain.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.logger import get_logger

logger = get_logger(__name__)

_Base = declarative_base()


class _LineageEventRecord(_Base):
    """Internal ORM model for the lineage_events table."""

    __tablename__ = "lineage_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False, index=True)
    dataset_id = Column(String(36), default="", index=True)
    dataset_name = Column(String(512), default="")
    stage = Column(String(64), nullable=False, index=True)
    event_type = Column(String(64), default="")
    column_name = Column(String(512), default="", index=True)
    summary = Column(Text, default="")
    detail = Column(Text, default="{}")  # JSON-encoded dict
    sequence = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<_LineageEventRecord(stage='{self.stage}', column='{self.column_name}')>"


@dataclass
class LineageEvent:
    """One recorded thing that happened during a pipeline run."""

    stage: str
    event_type: str = ""
    column: Optional[str] = None
    summary: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return the event as a JSON-serialisable dictionary."""
        return {
            "stage": self.stage,
            "event_type": self.event_type,
            "column": self.column,
            "summary": self.summary,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
        }


@dataclass
class LineageTrace:
    """Everything recorded for one pipeline run."""

    run_id: str = ""
    dataset_id: str = ""
    dataset_name: str = ""
    events: List[LineageEvent] = field(default_factory=list)

    @property
    def stage_names(self) -> List[str]:
        """Stages in the order they first appear."""
        seen: List[str] = []
        for event in self.events:
            if event.stage not in seen:
                seen.append(event.stage)
        return seen

    def stages(self) -> List[Dict[str, Any]]:
        """Summarise each stage, in execution order."""
        grouped: Dict[str, List[LineageEvent]] = {}
        for event in self.events:
            grouped.setdefault(event.stage, []).append(event)

        summaries = []
        for stage in self.stage_names:
            stage_events = grouped[stage]
            headline = next(
                (e for e in stage_events if e.column is None and e.summary), None
            )
            summaries.append({
                "stage": stage,
                "status": (headline.detail.get("status") if headline else None) or "completed",
                "event_count": len(stage_events),
                "summary": headline.summary if headline else f"{len(stage_events)} event(s)",
                "columns_touched": sorted({e.column for e in stage_events if e.column}),
            })
        return summaries

    def column_history(self, column: str) -> List[LineageEvent]:
        """Return everything that happened to one column, in order."""
        return [e for e in self.events if e.column == column]

    def columns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return per-column histories keyed by column name."""
        histories: Dict[str, List[Dict[str, Any]]] = {}
        for event in self.events:
            if event.column:
                histories.setdefault(event.column, []).append(event.to_dict())
        return histories

    def flow(self) -> str:
        """Render the run as an arrow chain: ``Raw -> Ingestion -> ...``."""
        return " -> ".join(["Raw Dataset"] + [s.title() for s in self.stage_names])

    def to_dict(self) -> Dict[str, Any]:
        """Return the trace as a JSON-serialisable dictionary."""
        return {
            "run_id": self.run_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "flow": self.flow(),
            "stages": self.stages(),
            "columns": self.columns(),
            "events": [e.to_dict() for e in self.events],
        }

    def column_frame(self, column: str):
        """Return one column's history as a DataFrame."""
        import pandas as pd

        history = self.column_history(column)
        if not history:
            return pd.DataFrame(columns=["stage", "event_type", "summary"])
        return pd.DataFrame([
            {"stage": e.stage, "event_type": e.event_type, "summary": e.summary}
            for e in history
        ])


class LineageTracker:
    """Records and retrieves pipeline lineage events.

    Parameters
    ----------
    db_path : str | None
        SQLite database path. Defaults to the registry's database so lineage
        and dataset metadata stay together.
    run_id : str | None
        Identifier for the current run. Generated when omitted.
    dataset_id, dataset_name : str
        Attached to every event recorded by this tracker.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        run_id: Optional[str] = None,
        dataset_id: str = "",
        dataset_name: str = "",
    ) -> None:
        self._db_path = db_path or "data/audithub.db"
        self._engine = create_engine(
            f"sqlite:///{self._db_path}",
            connect_args={"check_same_thread": False},
        )
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)

        self.run_id = run_id or str(uuid.uuid4())
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self._sequence = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        stage: str,
        summary: str,
        event_type: str = "",
        column: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a single lineage event.

        Failures are logged and swallowed: lineage is an audit trail, and a
        write problem must never take down the pipeline stage it is describing.
        """
        session = None
        try:
            # Inside the try: acquiring the session can fail too, and this
            # method promises never to raise into the stage it is describing.
            session = self._Session()
            self._sequence += 1
            session.add(_LineageEventRecord(
                run_id=self.run_id,
                dataset_id=self.dataset_id,
                dataset_name=self.dataset_name,
                stage=stage,
                event_type=event_type,
                column_name=column or "",
                summary=summary,
                detail=json.dumps(detail or {}, default=str),
                sequence=self._sequence,
            ))
            session.commit()
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            if session is not None:
                session.rollback()
            logger.warning("Could not record lineage event (%s/%s): %s", stage, column, exc)
        finally:
            if session is not None:
                session.close()

    def record_many(self, stage: str, events: List[Dict[str, Any]]) -> None:
        """Record several events for one stage.

        Each entry needs a ``summary`` and may carry ``column``, ``event_type``
        and ``detail``.
        """
        for event in events:
            self.record(
                stage=stage,
                summary=event.get("summary", ""),
                event_type=event.get("event_type", ""),
                column=event.get("column"),
                detail=event.get("detail"),
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def trace(self, run_id: Optional[str] = None) -> LineageTrace:
        """Return the full trace for a run (defaults to the current one)."""
        target = run_id or self.run_id
        session = self._Session()
        try:
            records = (
                session.query(_LineageEventRecord)
                .filter(_LineageEventRecord.run_id == target)
                .order_by(_LineageEventRecord.sequence, _LineageEventRecord.id)
                .all()
            )
            return self._build_trace(target, records)
        finally:
            session.close()

    def trace_for_dataset(self, dataset_id: str) -> LineageTrace:
        """Return the most recent run's trace for a given dataset."""
        session = self._Session()
        try:
            latest = (
                session.query(_LineageEventRecord)
                .filter(_LineageEventRecord.dataset_id == dataset_id)
                .order_by(_LineageEventRecord.id.desc())
                .first()
            )
            if latest is None:
                return LineageTrace(dataset_id=dataset_id)
            records = (
                session.query(_LineageEventRecord)
                .filter(_LineageEventRecord.run_id == latest.run_id)
                .order_by(_LineageEventRecord.sequence, _LineageEventRecord.id)
                .all()
            )
            return self._build_trace(latest.run_id, records)
        finally:
            session.close()

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recorded runs, most recent first."""
        session = self._Session()
        try:
            # Ascending, so each run's stages come out in execution order.
            records = (
                session.query(_LineageEventRecord)
                .order_by(_LineageEventRecord.id)
                .all()
            )
            runs: Dict[str, Dict[str, Any]] = {}
            for record in records:
                entry = runs.setdefault(record.run_id, {
                    "run_id": record.run_id,
                    "dataset_id": record.dataset_id,
                    "dataset_name": record.dataset_name,
                    "events": 0,
                    "stages": [],
                    "started_at": record.created_at.isoformat() if record.created_at else "",
                })
                entry["events"] += 1
                if record.stage not in entry["stages"]:
                    entry["stages"].append(record.stage)

            # Most recent run first.
            return list(reversed(list(runs.values())))[:limit]
        finally:
            session.close()

    @staticmethod
    def _build_trace(run_id: str, records: List[_LineageEventRecord]) -> LineageTrace:
        """Convert ORM rows into a :class:`LineageTrace`."""
        trace = LineageTrace(run_id=run_id)
        for record in records:
            trace.dataset_id = trace.dataset_id or (record.dataset_id or "")
            trace.dataset_name = trace.dataset_name or (record.dataset_name or "")
            try:
                detail = json.loads(record.detail) if record.detail else {}
            except json.JSONDecodeError:
                detail = {}
            trace.events.append(LineageEvent(
                stage=record.stage,
                event_type=record.event_type or "",
                column=record.column_name or None,
                summary=record.summary or "",
                detail=detail,
                timestamp=record.created_at.isoformat() if record.created_at else "",
                sequence=record.sequence or 0,
            ))
        return trace


__all__ = [
    "LineageEvent",
    "LineageTrace",
    "LineageTracker",
]
