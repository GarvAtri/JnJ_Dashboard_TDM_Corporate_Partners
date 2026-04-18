from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.agent.email import email_tool, format_daily_report_body
from src.agent.graph import run_send_email_report_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fixed_reference_date(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("ANALYTICS_REFERENCE_DATE", "2026-04-06")


def test_daily_summary_format_contains_required_sections() -> None:
	metrics = {
		"total_po_volume": 31,
		"average_variance": 497.13,
		"active_risk_count": 79,
		"addressable_spend_pct": 50.35,
		"total_spend": 1234567.89,
	}
	grouped = {
		"variance_bar": pd.DataFrame(
			[
				{"Root_Cause": "Price Variance", "Variance_vs_Budget": 7527679.87},
			],
		),
		"aging_histogram": pd.DataFrame(
			[
				{"Risk_ID": "1", "Days_Open": 400},
				{"Risk_ID": "2", "Days_Open": 25},
			],
		),
	}
	body = format_daily_report_body(metrics, grouped, date(2026, 4, 6), {"sector": "All"})

	assert "Executive Summary" in body
	assert "Spend and Variance Insights" in body
	assert "Risk and Governance Insights" in body
	assert "Saved Chat Responses" not in body


def test_email_tool_constructs_mailto_url(monkeypatch: pytest.MonkeyPatch) -> None:
	opened: list[str] = []
	monkeypatch.setattr("src.agent.email.webbrowser.open", lambda url: opened.append(url))
	result = email_tool.send_report(
		to="analytics-team@example.com",
		subject="Daily Analytics Report — 2026-04-06",
		body="Executive Summary\n- Total PO Volume: 31",
	)

	assert result["status"] == "success"
	assert opened
	assert opened[0].startswith("mailto:analytics-team%40example.com")
	assert "subject=" in opened[0]
	assert "body=" in opened[0]


def test_send_email_report_workflow_returns_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
	opened: list[str] = []
	monkeypatch.setattr("src.agent.email.webbrowser.open", lambda url: opened.append(url))

	result = run_send_email_report_workflow(filters={}, project_root=PROJECT_ROOT)

	assert result["status"] == "success"
	assert "email_report_result" in result
	assert result["email_report_result"]["to"] == "analytics-team@example.com"
	assert result["email_report_result"]["subject"].startswith("Daily Analytics Report —")
	assert "daily_report_summary" in result
	assert opened
