from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from src.agent.graph import run_dashboard_workflow, run_email_workflow, run_export_workflow
from src.agent.llm import ClaudeVarianceExplainer
from src.agent.workflow import compute_metrics_node, load_data_node


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def fixed_reference_date(monkeypatch: pytest.MonkeyPatch) -> None:
	monkeypatch.setenv("ANALYTICS_REFERENCE_DATE", "2026-04-06")


def test_dashboard_workflow_returns_expected_metrics() -> None:
	result = run_dashboard_workflow(project_root=PROJECT_ROOT)

	assert result["status"] == "success"
	metrics = result["dashboard_state"]["metrics"]
	assert metrics["total_po_volume"] == 31
	assert metrics["average_variance"] == pytest.approx(497.13, abs=0.01)
	assert metrics["active_risk_count"] == 79
	assert metrics["addressable_spend_pct"] == pytest.approx(50.35, abs=0.01)


def test_risk_aging_logic_and_heatmap_derivation_are_stable() -> None:
	loaded = load_data_node({"config": {"project_root": str(PROJECT_ROOT)}})
	risk = loaded["risk"]

	assert risk["Days_Open"].min() >= 0
	assert set(risk["Impact_Score"].unique()) <= {1, 2, 3, 5}
	assert set(risk["Likelihood_Score"].unique()) <= {1, 2, 3, 4, 5}
	assert int(risk.loc[risk["Active"], "Risk_ID"].count()) == 79


def test_variance_drift_logic_handles_single_month_data() -> None:
	loaded = load_data_node({"config": {"project_root": str(PROJECT_ROOT)}})
	computed = compute_metrics_node(loaded)

	drift = computed["grouped"]["variance_drift"]
	assert drift["direction"] == "stable"
	assert drift["delta"] == 0.0


def test_email_and_export_actions_return_structured_results(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	opened_urls: list[str] = []
	monkeypatch.setattr("src.agent.tools.webbrowser.open", lambda url: opened_urls.append(url))
	monkeypatch.setenv("EXPORT_DIR", str(tmp_path))

	dashboard = run_dashboard_workflow(project_root=PROJECT_ROOT)
	risk_id = dashboard["dashboard_state"]["tables"]["open_risks"][0]["risk_id"]

	email_result = run_email_workflow({}, risk_id, project_root=PROJECT_ROOT)
	export_result = run_export_workflow({}, project_root=PROJECT_ROOT)

	assert email_result["email_result"]["status"] == "success"
	assert "recipient" in email_result["email_result"]
	assert opened_urls and opened_urls[0].startswith("mailto:")
	assert export_result["export_result"]["status"] == "success"
	assert Path(export_result["export_result"]["path"]).exists()


def test_workflow_shape_matches_expected_contract() -> None:
	expected = json.loads((PROJECT_ROOT / "tests" / "expected" / "sample_output.json").read_text())
	result = run_dashboard_workflow(project_root=PROJECT_ROOT)
	dashboard_state = result["dashboard_state"]

	assert sorted(dashboard_state.keys()) == sorted(expected["dashboard_state_keys"])
	assert sorted(dashboard_state["metrics"].keys()) == sorted(expected["metric_keys"])


def test_variance_explanations_are_two_sentences() -> None:
	explainer = ClaudeVarianceExplainer(api_key=None)
	message = explainer.explain_variance(
		{
			"root_cause": "Price Variance",
			"variance_total": 7527679.87,
			"variance_drift": {"direction": "stable", "delta": 0.0},
			"concentration_risk": {"top_share": 0.51},
		}
	)["message"]
	sentences = [segment for segment in re.split(r"(?<=[.!?])\s+", message.strip()) if segment]
	assert len(sentences) == 2
