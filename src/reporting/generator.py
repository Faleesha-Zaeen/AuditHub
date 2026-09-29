"""
AuditHub Reporting - Report Engine
====================================

Generates premium aggregated reports in HTML, JSON, CSV, and PDF-ready formats.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.constants import REPORTS_DIR
from src.utils.logger import get_logger
from src.utils.helpers import ensure_directory_exists, unique_filename

logger = get_logger(__name__)

# Status -> badge colour, shared by the drift, version and explainability blocks.
_STATUS_COLORS: Dict[str, str] = {
    "CRITICAL": "#ef4444",
    "WARNING": "#f59e0b",
    "STABLE": "#10b981",
    "NEW_COLUMN": "#3b82f6",
    "REMOVED_COLUMN": "#ef4444",
    "TYPE_CHANGED": "#8b5cf6",
    "NOT_COMPARABLE": "#6b7280",
    "ERROR": "#ef4444",
    "INFO": "#3b82f6",
}


def _badge(status: str) -> str:
    """Render a coloured status badge."""
    color = _STATUS_COLORS.get(str(status).upper(), "#6b7280")
    return f'<span class="badge" style="background-color: {color};">{status}</span>'


def _bar_pair(label: str, reference_pct: float, current_pct: float) -> str:
    """Render one reference-vs-current comparison bar.

    Drawn with plain divs rather than an embedded image so the report stays a
    single self-contained HTML file with no external assets.
    """
    ref_w = max(0.0, min(100.0, float(reference_pct)))
    cur_w = max(0.0, min(100.0, float(current_pct)))
    return f"""
    <div style="margin-bottom: 10px;">
        <div style="font-size: 0.78rem; color: #374151; margin-bottom: 3px;">
            <code>{label}</code>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 2px;">
            <span style="font-size: 0.7rem; color: #6b7280; width: 62px;">reference</span>
            <div style="flex: 1; background: #f3f4f6; border-radius: 3px; height: 12px;">
                <div style="width: {ref_w}%; background: #93c5fd; height: 12px; border-radius: 3px;"></div>
            </div>
            <span style="font-size: 0.7rem; color: #6b7280; width: 48px; text-align: right;">{reference_pct:.1f}%</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 0.7rem; color: #6b7280; width: 62px;">current</span>
            <div style="flex: 1; background: #f3f4f6; border-radius: 3px; height: 12px;">
                <div style="width: {cur_w}%; background: #f59e0b; height: 12px; border-radius: 3px;"></div>
            </div>
            <span style="font-size: 0.7rem; color: #6b7280; width: 48px; text-align: right;">{current_pct:.1f}%</span>
        </div>
    </div>
    """


class ReportGenerator:
    """Consolidates metrics, validation results, repair history, and ML metrics into styled reports."""

    def __init__(self) -> None:
        """Initialize the ReportGenerator."""
        ensure_directory_exists(REPORTS_DIR)

    # ------------------------------------------------------------------
    # Optional section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _drift_block(drift_report: Optional[Dict[str, Any]]) -> str:
        """Render the data drift section."""
        if not drift_report:
            return ""

        status = drift_report.get("status", "STABLE")
        columns = drift_report.get("columns", [])

        rows = ""
        for col in columns:
            psi = col.get("psi")
            p_value = col.get("p_value")
            rows += f"""
            <tr>
                <td>{_badge(col.get('status', ''))}</td>
                <td><code>{col.get('column')}</code></td>
                <td>{col.get('column_kind', '-')}</td>
                <td>{'-' if psi is None else f'{psi:.4f}'}</td>
                <td>{'-' if p_value is None else f'{p_value:.3g}'}</td>
                <td style="font-size: 0.82rem;">{col.get('explanation', '')}</td>
            </tr>
            """
        if not rows:
            rows = "<tr><td colspan='6' style='text-align:center;color:#6b7280;'>No columns compared.</td></tr>"

        # Distribution bars for the worst offenders.
        charts = ""
        drifted = [c for c in columns if c.get("status") in ("WARNING", "CRITICAL")][:4]
        for col in drifted:
            bars = "".join(
                _bar_pair(entry["bin"], entry["reference_pct"], entry["current_pct"])
                for entry in col.get("distribution", [])[:12]
            )
            if bars:
                charts += f"""
                <div style="margin-bottom: 18px;">
                    <h4 style="margin: 6px 0;">
                        <code>{col.get('column')}</code> {_badge(col.get('status', ''))}
                        <span style="font-weight: 400; color:#6b7280; font-size:0.8rem;">
                            PSI {col.get('psi'):.4f}
                        </span>
                    </h4>
                    {bars}
                </div>
                """

        chart_section = (
            f"<div class='card'><h3>Distribution Shifts</h3>{charts}</div>" if charts else ""
        )

        return f"""
        <div class="card">
            <h3>🌊 Data Drift — {drift_report.get('reference_name', 'reference')} vs {drift_report.get('current_name', 'current')}</h3>
            <p>
                Overall verdict: {_badge(status)}
                &nbsp;·&nbsp; mean PSI <strong>{drift_report.get('overall_score', 0.0):.4f}</strong>
                &nbsp;·&nbsp; highest column PSI <strong>{drift_report.get('max_score', 0.0):.4f}</strong>
            </p>
            <p style="color:#6b7280; font-size:0.85rem;">
                {drift_report.get('summary', '')}
                Reference rows: {drift_report.get('reference_rows', 0):,} ·
                Current rows: {drift_report.get('current_rows', 0):,}
            </p>
            <table>
                <thead>
                    <tr>
                        <th>Status</th><th>Column</th><th>Kind</th>
                        <th>PSI</th><th>p-value</th><th>What changed</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        {chart_section}
        """

    @staticmethod
    def _version_block(version_comparison: Optional[Dict[str, Any]]) -> str:
        """Render the dataset version comparison section."""
        if not version_comparison:
            return ""

        vc = version_comparison
        changes = vc.get("column_changes", [])
        rows = ""
        for change in changes:
            rows += f"""
            <tr>
                <td>{_badge(change.get('severity', 'INFO'))}</td>
                <td><code>{change.get('column')}</code></td>
                <td>{change.get('change_type')}</td>
                <td style="font-size: 0.82rem;">{change.get('detail', '')}</td>
            </tr>
            """
        if not rows:
            rows = "<tr><td colspan='4' style='text-align:center;color:#6b7280;'>No column-level changes detected.</td></tr>"

        def _delta(label: str, before: Any, after: Any) -> str:
            return f"""
            <div class="metric-item">
                <span class="metric-key">{label}</span>
                <span class="metric-value">{before} → {after}</span>
            </div>
            """

        return f"""
        <div class="card">
            <h3>🧬 Version Comparison — {vc.get('left_name', 'V1')} → {vc.get('right_name', 'V2')}</h3>
            <div class="metric-grid">
                {_delta('Rows', f"{vc.get('left_rows', 0):,}", f"{vc.get('right_rows', 0):,}")}
                {_delta('Columns', vc.get('left_columns', 0), vc.get('right_columns', 0))}
                {_delta('Missing %', f"{vc.get('left_missing_pct', 0.0):.2f}%", f"{vc.get('right_missing_pct', 0.0):.2f}%")}
                {_delta('Duplicates', vc.get('left_duplicates', 0), vc.get('right_duplicates', 0))}
            </div>
            <p style="color:#6b7280; font-size:0.85rem;">{vc.get('summary', '')}</p>
            <table>
                <thead>
                    <tr><th>Severity</th><th>Column</th><th>Change</th><th>Detail</th></tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    @staticmethod
    def _lineage_block(lineage: Optional[Dict[str, Any]]) -> str:
        """Render the pipeline lineage section."""
        if not lineage:
            return ""

        stages = lineage.get("stages", [])
        chain = " → ".join(
            f"<span style='background:#eef2ff;padding:3px 9px;border-radius:12px;"
            f"font-size:0.8rem;'>{s.get('stage')}</span>"
            for s in stages
        )

        stage_rows = ""
        for stage in stages:
            stage_rows += f"""
            <tr>
                <td><code>{stage.get('stage')}</code></td>
                <td>{stage.get('status', '-')}</td>
                <td>{stage.get('event_count', 0)}</td>
                <td style="font-size:0.82rem;">{stage.get('summary', '')}</td>
            </tr>
            """
        if not stage_rows:
            stage_rows = "<tr><td colspan='4' style='text-align:center;color:#6b7280;'>No lineage recorded.</td></tr>"

        column_rows = ""
        for name, events in (lineage.get("columns") or {}).items():
            steps = " → ".join(e.get("summary", "") for e in events)
            column_rows += f"""
            <tr><td><code>{name}</code></td><td style="font-size:0.82rem;">{steps}</td></tr>
            """
        column_section = ""
        if column_rows:
            column_section = f"""
            <h4 style="margin-top:16px;">Column histories</h4>
            <table>
                <thead><tr><th>Column</th><th>What happened through the pipeline</th></tr></thead>
                <tbody>{column_rows}</tbody>
            </table>
            """

        return f"""
        <div class="card">
            <h3>🔗 Data Lineage</h3>
            <p style="line-height: 2.2;">{chain}</p>
            <table>
                <thead><tr><th>Stage</th><th>Status</th><th>Events</th><th>Summary</th></tr></thead>
                <tbody>{stage_rows}</tbody>
            </table>
            {column_section}
        </div>
        """

    @staticmethod
    def _explainability_block(explainability: Optional[Dict[str, Any]]) -> str:
        """Render the model explainability section."""
        if not explainability:
            return ""

        global_exp = explainability.get("global_importance", [])
        method = explainability.get("method", "unknown")

        max_importance = max((abs(f.get("importance", 0.0)) for f in global_exp), default=1.0) or 1.0
        global_rows = ""
        for rank, feat in enumerate(global_exp[:20], start=1):
            width = abs(feat.get("importance", 0.0)) / max_importance * 100
            global_rows += f"""
            <tr>
                <td>{rank}</td>
                <td><code>{feat.get('feature')}</code></td>
                <td style="width: 45%;">
                    <div style="background:#f3f4f6;border-radius:3px;height:12px;">
                        <div style="width:{width:.1f}%;background:#6366f1;height:12px;border-radius:3px;"></div>
                    </div>
                </td>
                <td>{feat.get('importance', 0.0):.6f}</td>
            </tr>
            """
        if not global_rows:
            global_rows = "<tr><td colspan='4' style='text-align:center;color:#6b7280;'>No global importance available.</td></tr>"

        local_section = ""
        for local in explainability.get("local_explanations", [])[:3]:
            contrib_rows = ""
            for c in local.get("contributions", [])[:10]:
                direction = c.get("direction", "")
                color = "#ef4444" if direction == "increases" else "#10b981"
                contrib_rows += f"""
                <tr>
                    <td><code>{c.get('feature')}</code></td>
                    <td>{c.get('value')}</td>
                    <td style="color:{color};font-weight:600;">{direction}</td>
                    <td>{c.get('contribution', 0.0):+.6f}</td>
                </tr>
                """
            local_section += f"""
            <div style="margin-top: 14px;">
                <h4>Row {local.get('row_index')} — predicted
                    <code>{local.get('prediction')}</code>
                    {f"(confidence {local.get('confidence'):.3f})" if local.get('confidence') is not None else ""}
                </h4>
                <table>
                    <thead><tr><th>Feature</th><th>Value</th><th>Effect</th><th>Contribution</th></tr></thead>
                    <tbody>{contrib_rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="card">
            <h3>🧠 Model Explainability</h3>
            <p style="color:#6b7280; font-size:0.85rem;">
                Method: <code>{method}</code>.
                {explainability.get('summary', '')}
            </p>
            <h4>Global feature importance</h4>
            <table>
                <thead><tr><th>#</th><th>Feature</th><th></th><th>Importance</th></tr></thead>
                <tbody>{global_rows}</tbody>
            </table>
            {local_section}
        </div>
        """

    def generate_html(
        self,
        dataset_name: str,
        validation_report: Dict[str, Any],
        health_report: Dict[str, Any],
        audit_report: Dict[str, Any],
        repair_log: List[Dict[str, Any]],
        robustness_results: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
        version_comparison: Optional[Dict[str, Any]] = None,
        lineage: Optional[Dict[str, Any]] = None,
        explainability: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render a premium HTML report consolidation.

        Parameters
        ----------
        dataset_name : str
            Name of the audited dataset.
        validation_report : dict
            Dict representation of ValidationReport.
        health_report : dict
            Dict representation of DatasetHealthReport.
        audit_report : dict
            Dict representation of QualityAuditReport.
        repair_log : list
            History log from DatasetRepairer.
        robustness_results : dict | None
            Output from RobustnessEvaluator.
        training_metrics : dict | None
            Evaluation output from MLTrainingEngine.
        drift_report : dict | None
            Dict representation of a ``DriftReport``.
        version_comparison : dict | None
            Dict representation of a ``VersionComparison``.
        lineage : dict | None
            Dict representation of a ``LineageTrace``.
        explainability : dict | None
            Dict representation of an ``ExplanationReport``.

        Returns
        -------
        str
            Sleek, styled HTML document.
        """
        logger.info("Generating HTML quality audit report for %s", dataset_name)
        
        # Color palettes & score indicators
        score = health_report.get("overall_score", 0.0)
        grade = health_report.get("grade", "F")
        score_color = "#10b981"  # green
        if score < 40:
            score_color = "#ef4444"  # red
        elif score < 75:
            score_color = "#f59e0b"  # amber
        elif score < 90:
            score_color = "#3b82f6"  # blue

        # Generate findings HTML rows
        findings_rows = ""
        findings = audit_report.get("findings", [])
        for f in findings:
            sev = f.get("severity", "INFO")
            badge_color = "#6b7280"  # gray
            if sev == "ERROR":
                badge_color = "#ef4444"
            elif sev == "WARNING":
                badge_color = "#f59e0b"
            elif sev == "INFO":
                badge_color = "#3b82f6"

            findings_rows += f"""
            <tr>
                <td><span class="badge" style="background-color: {badge_color};">{sev}</span></td>
                <td><strong>{f.get('dimension')}</strong></td>
                <td><code>{f.get('column') or 'Dataset-wide'}</code></td>
                <td>{f.get('message')}</td>
            </tr>
            """
        
        if not findings_rows:
            findings_rows = "<tr><td colspan='4' style='text-align: center; color: #6b7280;'>No anomalies detected in the dataset.</td></tr>"

        # Generate repair log HTML rows
        repair_rows = ""
        for r in repair_log:
            repair_rows += f"""
            <tr>
                <td>{r.get('step')}</td>
                <td><code>{r.get('action')}</code></td>
                <td>{json.dumps(r.get('details', {}))}</td>
                <td><small style="color: #6b7280;">{r.get('timestamp')}</small></td>
            </tr>
            """
        if not repair_rows:
            repair_rows = "<tr><td colspan='4' style='text-align: center; color: #6b7280;'>No repairs applied.</td></tr>"

        # ML metrics block
        ml_block = "<div class='card'><p style='color: #6b7280; text-align: center;'>No ML training results available in this report run.</p></div>"
        if training_metrics:
            ml_rows = ""
            for k, v in training_metrics.items():
                if k != "feature_importances":
                    ml_rows += f"""
                    <div class="metric-item">
                        <span class="metric-key">{k.upper()}</span>
                        <span class="metric-val">{v if isinstance(v, str) else f"{v:.4f}"}</span>
                    </div>
                    """
            ml_block = f"""
            <div class="card">
                <h3>🤖 Machine Learning Model Performance</h3>
                <div class="metrics-grid">
                    {ml_rows}
                </div>
            </div>
            """

        # Robustness block
        robustness_block = ""
        if robustness_results:
            rob_rows = ""
            levels = robustness_results.get("levels", [])
            perfs = robustness_results.get("performances", [])
            for lvl, perf in zip(levels, perfs):
                rob_rows += f"""
                <tr>
                    <td><code>{lvl * 100:.0f}%</code></td>
                    <td>{perf:.4f}</td>
                </tr>
                """
            robustness_block = f"""
            <div class="card">
                <h3>💪 Robustness Decay Curve (Mutation: {robustness_results.get('mutation_type')})</h3>
                <p>Dataset Elasticity Score: <strong>{robustness_results.get('elasticity_score'):.4f}</strong></p>
                <table>
                    <thead>
                        <tr>
                            <th>Anomaly Intensity</th>
                            <th>Model Metric ({robustness_results.get('metric_name')})</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rob_rows}
                    </tbody>
                </table>
            </div>
            """

        # Optional sections. Each renders to an empty string when its data was
        # not produced by this run, so the report stays valid either way.
        drift_block = self._drift_block(drift_report)
        version_block = self._version_block(version_comparison)
        lineage_block = self._lineage_block(lineage)
        explainability_block = self._explainability_block(explainability)

        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>AuditHub Dataset Quality Audit Report</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
                body {{
                    font-family: 'Inter', sans-serif;
                    background-color: #0f172a;
                    color: #e2e8f0;
                    margin: 0;
                    padding: 40px;
                    line-height: 1.6;
                }}
                .container {{
                    max-width: 1000px;
                    margin: 0 auto;
                }}
                header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 1px solid #334155;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                h1 {{
                    font-size: 2.2rem;
                    margin: 0;
                    background: linear-gradient(135deg, #60a5fa, #3b82f6);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                h2, h3 {{
                    color: #94a3b8;
                }}
                .score-container {{
                    display: flex;
                    align-items: center;
                    background-color: #1e293b;
                    padding: 15px 25px;
                    border-radius: 12px;
                    border: 1px solid #334155;
                }}
                .score-circle {{
                    width: 70px;
                    height: 70px;
                    border-radius: 50%;
                    background-color: #0f172a;
                    border: 4px solid {score_color};
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    font-size: 1.4rem;
                    font-weight: 700;
                    color: {score_color};
                    margin-right: 15px;
                }}
                .grade-box {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: #fff;
                    background-color: {score_color};
                    padding: 5px 15px;
                    border-radius: 8px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .card {{
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 12px;
                    padding: 24px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #334155;
                }}
                th {{
                    background-color: #0f172a;
                    color: #94a3b8;
                    font-weight: 600;
                }}
                tr:hover {{
                    background-color: #334155;
                }}
                .badge {{
                    padding: 4px 8px;
                    border-radius: 6px;
                    font-size: 0.75rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    color: white;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric-item {{
                    background-color: #0f172a;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px solid #334155;
                    text-align: center;
                }}
                .metric-key {{
                    display: block;
                    font-size: 0.75rem;
                    color: #6b7280;
                    margin-bottom: 5px;
                    font-weight: 600;
                }}
                .metric-val {{
                    font-size: 1.3rem;
                    font-weight: 700;
                    color: #60a5fa;
                }}
                .deductions-list {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                .deductions-list li {{
                    padding: 8px 12px;
                    background-color: #0f172a;
                    margin-bottom: 6px;
                    border-radius: 6px;
                    border-left: 3px solid #ef4444;
                    font-size: 0.9rem;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <div>
                        <h1>📊 AuditHub Quality Audit Report</h1>
                        <p style="margin: 5px 0 0 0; color: #6b7280;">Dataset: <strong>{dataset_name}</strong> | Generated on: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                    </div>
                    <div class="score-container">
                        <div class="score-circle">{score:.0f}</div>
                        <div>
                            <span style="display:block; font-size: 0.75rem; color: #6b7280; font-weight: 600; text-transform: uppercase;">Overall Health</span>
                            <span class="grade-box">Grade {grade}</span>
                        </div>
                    </div>
                </header>

                <div class="grid">
                    <div class="card">
                        <h3>❤️ Dataset Health Score Breakdown</h3>
                        <div class="metrics-grid">
                            <div class="metric-item">
                                <span class="metric-key">COMPLETENESS</span>
                                <span class="metric-val">{health_report.get('dimensions', {}).get('completeness', {}).get('score', 100.0):.0f}%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-key">UNIQUENESS</span>
                                <span class="metric-val">{health_report.get('dimensions', {}).get('uniqueness', {}).get('score', 100.0):.0f}%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-key">CONSISTENCY</span>
                                <span class="metric-val">{health_report.get('dimensions', {}).get('consistency', {}).get('score', 100.0):.0f}%</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-key">ACCURACY</span>
                                <span class="metric-val">{health_report.get('dimensions', {}).get('accuracy', {}).get('score', 100.0):.0f}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h3>⚠️ Score Deduction Explanations</h3>
                        <ul class="deductions-list">
                            {"".join(f"<li>{d}</li>" for d in health_report.get('explanations', [])[:5])}
                            {"" if len(health_report.get('explanations', [])) <= 5 else f"<li style='border-left-color: #3b82f6; font-style: italic;'>+ {len(health_report.get('explanations', [])) - 5} more deductions.</li>"}
                            {"<li style='border-left-color: #10b981;'>Perfect Score! No deductions registered.</li>" if not health_report.get('explanations', []) else ""}
                        </ul>
                    </div>
                </div>

                <div class="card" style="margin-bottom: 30px;">
                    <h3>🔍 Quality Audit Anomalies</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Severity</th>
                                <th>Quality Dimension</th>
                                <th>Column/Scope</th>
                                <th>Observation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {findings_rows}
                        </tbody>
                    </table>
                </div>

                <div class="grid">
                    {ml_block}
                    {robustness_block}
                    {drift_block}
                    {version_block}
                    {lineage_block}
                    {explainability_block}
                </div>

                <div class="card">
                    <h3>🔧 Applied Modifications & Repairs</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>Repair Operation</th>
                                <th>Action details</th>
                                <th>Timestamp</th>
                            </tr>
                        </thead>
                        <tbody>
                            {repair_rows}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return html_template

    def export_report(
        self,
        dataset_name: str,
        validation_report: Dict[str, Any],
        health_report: Dict[str, Any],
        audit_report: Dict[str, Any],
        repair_log: List[Dict[str, Any]],
        robustness_results: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, Any]] = None,
        format_type: str = "html",
        drift_report: Optional[Dict[str, Any]] = None,
        version_comparison: Optional[Dict[str, Any]] = None,
        lineage: Optional[Dict[str, Any]] = None,
        explainability: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Compile and save a quality audit report to REPORTS_DIR.

        Parameters
        ----------
        dataset_name : str
            Name of audited dataset.
        validation_report : dict
            Validation report.
        health_report : dict
            Health report.
        audit_report : dict
            Audit report.
        repair_log : list
            Repair log.
        robustness_results : dict | None
            Robustness evaluation results.
        training_metrics : dict | None
            ML training comparison metrics.
        format_type : str
            Output file format: 'html' or 'json'.

        Returns
        -------
        Path
            Path to the saved report.
        """
        ensure_directory_exists(REPORTS_DIR)
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in dataset_name)

        if format_type.lower() == "html":
            content = self.generate_html(
                dataset_name,
                validation_report,
                health_report,
                audit_report,
                repair_log,
                robustness_results,
                training_metrics,
                drift_report=drift_report,
                version_comparison=version_comparison,
                lineage=lineage,
                explainability=explainability,
            )
            filename = unique_filename(f"{safe_name}_quality_report.html")
            filepath = REPORTS_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Saved HTML report to %s", filepath)
            return filepath
        
        elif format_type.lower() == "json":
            report_dict = {
                "dataset_name": dataset_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "validation": validation_report,
                "health": health_report,
                "audit": audit_report,
                "repairs": repair_log,
                "robustness": robustness_results,
                "training": training_metrics,
                "drift": drift_report,
                "version_comparison": version_comparison,
                "lineage": lineage,
                "explainability": explainability,
            }
            filename = unique_filename(f"{safe_name}_quality_report.json")
            filepath = REPORTS_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            logger.info("Saved JSON report to %s", filepath)
            return filepath

        else:
            raise ValueError(f"Unsupported report format: {format_type}")
