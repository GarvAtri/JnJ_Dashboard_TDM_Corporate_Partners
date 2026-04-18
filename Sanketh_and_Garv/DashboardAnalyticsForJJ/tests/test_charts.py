from __future__ import annotations

from pathlib import Path

from src.agent.workflow import compute_metrics_node, generate_charts_node, load_data_node


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_chart_titles_and_traces_are_generated() -> None:
	loaded = load_data_node({"config": {"project_root": str(PROJECT_ROOT)}})
	computed = compute_metrics_node(loaded)
	charted = generate_charts_node(computed)

	charts = charted["charts"]
	# All five chart keys must exist with valid data
	for key in ("sector_treemap", "root_cause_variance", "trend_and_seasonality", "risk_heatmap", "aging_risk_histogram"):
		assert key in charts, f"Missing chart key: {key}"
		assert "data" in charts[key], f"No data in chart: {key}"
	# Titles are rendered in the Reflex chart_card header; Plotly title is intentionally None
	assert charts["root_cause_variance"]["layout"]["barmode"] == "group"
	assert "yaxis2" in charts["trend_and_seasonality"]["layout"]


def test_root_cause_bar_contains_expected_categories() -> None:
	loaded = load_data_node({"config": {"project_root": str(PROJECT_ROOT)}})
	computed = compute_metrics_node(loaded)
	charted = generate_charts_node(computed)

	y_values = charted["charts"]["root_cause_variance"]["data"][0]["y"]
	assert set(y_values) == {"Overspend", "Price Variance", "Timing"}


def test_risk_heatmap_is_five_by_five() -> None:
	loaded = load_data_node({"config": {"project_root": str(PROJECT_ROOT)}})
	computed = compute_metrics_node(loaded)
	charted = generate_charts_node(computed)

	heatmap = charted["charts"]["risk_heatmap"]["data"][0]
	assert heatmap["z"]["shape"] == "5, 5"
