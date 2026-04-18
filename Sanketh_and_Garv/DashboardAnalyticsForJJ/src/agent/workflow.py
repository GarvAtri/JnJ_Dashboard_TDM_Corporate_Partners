"""Deterministic workflow nodes and orchestration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, TypedDict
import os

from dotenv import load_dotenv
import pandas as pd

from src.agent.email import email_tool, format_daily_report_body
from src.agent.llm import ClaudeVarianceExplainer
from src.agent.tools import export_dashboard_pdf, open_risk_owner_email
from src.ui.charts import (
	build_aging_risk_histogram,
	build_risk_heatmap,
	build_root_cause_variance_bar,
	build_sector_treemap,
	build_trend_and_seasonality_line,
)


class WorkflowState(TypedDict, total=False):
	"""State contract shared across LangGraph nodes."""

	config: dict[str, Any]
	filters: dict[str, str]
	spend_header: pd.DataFrame
	spend_detail: pd.DataFrame
	risk: pd.DataFrame
	metrics: dict[str, Any]
	grouped: dict[str, Any]
	filter_options: dict[str, list[str]]
	charts: dict[str, dict[str, Any]]
	tables: dict[str, list[dict[str, Any]]]
	dashboard_state: dict[str, Any]
	explanation: dict[str, Any]
	email_result: dict[str, Any]
	email_report_result: dict[str, Any]
	daily_report_summary: str
	export_result: dict[str, Any]
	status: str
	errors: list[str]


@dataclass(frozen=True)
class DashboardConfig:
	"""Resolved runtime configuration for the dashboard."""

	project_root: Path
	workspace_root: Path
	spend_file_path: Path
	risk_file_path: Path
	export_dir: Path
	anthropic_api_key: str | None
	analysis_date: date


REQUIRED_SPEND_HEADER_COLUMNS = {
	"PO_Number",
	"PO_Status",
	"Business_Sector",
	"Addressable_Flag",
	"PO_Total_Amount",
	"Last_Updated_Timestamp",
}
REQUIRED_SPEND_DETAIL_COLUMNS = {
	"PO_Number",
	"Sector",
	"Spend_Amount",
	"Variance_vs_Budget",
	"Root_Cause_Code",
	"Last_Updated_Timestamp",
}
REQUIRED_RISK_COLUMNS = {
	"Risk #",
	"Risk Description",
	"Risk Owner",
	"Risk Status",
	"Risk Category",
	"Risk Level",
	"Risk ERM Type",
	"Open Date",
	"Closed Date",
}
IMPACT_SCORE_MAP = {"Low": 2, "Medium": 3, "High": 5}


def load_config(project_root: Path | None = None) -> DashboardConfig:
	"""Load configuration from environment and project defaults."""
	resolved_project_root = project_root or Path(__file__).resolve().parents[2]
	load_dotenv(resolved_project_root / ".env")
	workspace_root = resolved_project_root.parent
	spend_file_path = Path(
		os.getenv(
			"SPEND_FILE_PATH",
			str(workspace_root / "Enterprise_Spend_Jan_2026.xlsx"),
		)
	)
	risk_file_path = Path(
		os.getenv(
			"RISK_FILE_PATH",
			str(workspace_root / "RiskRegisterSample.xlsx"),
		)
	)
	export_dir = Path(os.getenv("EXPORT_DIR", str(resolved_project_root / "Reports")))
	analysis_date_raw = os.getenv("ANALYTICS_REFERENCE_DATE")
	analysis_date = (
		datetime.strptime(analysis_date_raw, "%Y-%m-%d").date()
		if analysis_date_raw
		else date.today()
	)
	return DashboardConfig(
		project_root=resolved_project_root,
		workspace_root=workspace_root,
		spend_file_path=spend_file_path,
		risk_file_path=risk_file_path,
		export_dir=export_dir,
		anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
		analysis_date=analysis_date,
	)


def load_data_node(state: WorkflowState) -> WorkflowState:
	"""Load and normalize the spend and risk sources."""
	config = load_config(Path(state.get("config", {}).get("project_root")) if state.get("config") else None)
	spend_header = pd.read_excel(config.spend_file_path, sheet_name="PO_Header")
	spend_detail = pd.read_excel(config.spend_file_path, sheet_name="SpendDetails_JobAid")
	risk = pd.read_excel(config.risk_file_path)

	_validate_columns(spend_header, REQUIRED_SPEND_HEADER_COLUMNS, "PO_Header")
	_validate_columns(spend_detail, REQUIRED_SPEND_DETAIL_COLUMNS, "SpendDetails_JobAid")
	_validate_columns(risk, REQUIRED_RISK_COLUMNS, "RiskRegisterSample")

	normalized_header = _normalize_spend_header(spend_header)
	normalized_detail = _normalize_spend_detail(spend_detail)
	normalized_risk = _normalize_risk(risk, config.analysis_date)

	return {
		"config": _serialize_config(config),
		"spend_header": normalized_header,
		"spend_detail": normalized_detail,
		"risk": normalized_risk,
		"filters": state.get("filters", _default_filters()),
		"filter_options": _build_filter_options(normalized_header, normalized_risk),
		"status": "loaded",
	}


def compute_metrics_node(state: WorkflowState) -> WorkflowState:
	"""Compute executive metrics and grouped datasets."""
	filters = state.get("filters", _default_filters())
	spend_header = state["spend_header"]
	spend_detail = state["spend_detail"]
	risk = state["risk"]

	filtered_header, filtered_detail, filtered_risk = apply_dashboard_filters(
		spend_header,
		spend_detail,
		risk,
		filters,
	)

	total_spend = float(filtered_header["PO_Total_Amount"].sum())
	addressable_spend = float(
		filtered_header.loc[filtered_header["Addressable_Flag"].eq("Yes"), "PO_Total_Amount"].sum()
	)
	metrics = {
		"total_po_volume": int(filtered_header["PO_Number"].nunique()),
		"average_variance": round(float(filtered_detail["Variance_vs_Budget"].mean() or 0.0), 2),
		"active_risk_count": int(filtered_risk["Active"].sum()),
		"addressable_spend_pct": round((addressable_spend / total_spend) * 100, 2) if total_spend else 0.0,
		"total_spend": round(total_spend, 2),
	}

	grouped = {
		"sector_treemap": _group_sector_spend(filtered_header),
		"variance_bar": _group_root_cause_variance(filtered_detail),
		"trend_line": _build_trend_frame(filtered_header, filtered_detail),
		"risk_heatmap": _group_risk_heatmap(filtered_risk),
		"aging_histogram": filtered_risk.loc[filtered_risk["Active"], ["Risk_ID", "Days_Open"]].copy(),
		"open_risks": _build_open_risk_table(filtered_risk),
		"variance_drift": _compute_variance_drift(filtered_detail),
		"concentration_risk": _compute_concentration_risk(filtered_header),
	}

	tables = {
		"open_risks": grouped["open_risks"],
	}
	return {
		"metrics": metrics,
		"grouped": grouped,
		"tables": tables,
		"filters": filters,
		"filter_options": state.get("filter_options", {}),
		"status": "computed",
	}


def generate_charts_node(state: WorkflowState) -> WorkflowState:
	"""Create Plotly chart payloads from grouped analytics data."""
	grouped = state["grouped"]
	filters = state.get("filters", _default_filters())
	charts = {
		"sector_treemap": build_sector_treemap(grouped["sector_treemap"], filters).to_dict(),
		"root_cause_variance": build_root_cause_variance_bar(grouped["variance_bar"], filters).to_dict(),
		"trend_and_seasonality": build_trend_and_seasonality_line(grouped["trend_line"], filters).to_dict(),
		"risk_heatmap": build_risk_heatmap(grouped["risk_heatmap"], filters).to_dict(),
		"aging_risk_histogram": build_aging_risk_histogram(grouped["aging_histogram"], filters).to_dict(),
	}
	return {
		"charts": charts,
		"status": "charted",
	}


def update_state_node(state: WorkflowState) -> WorkflowState:
	"""Translate workflow outputs into a Reflex-friendly state payload."""
	metrics = state["metrics"]
	grouped = state["grouped"]
	dashboard_state = {
		"metrics": metrics,
		"charts": state["charts"],
		"tables": state["tables"],
		"filter_options": state.get("filter_options", {}),
		"filters": state.get("filters", _default_filters()),
		"insights": _build_summary_lines(metrics, grouped),
		"grouped": {
			"sector_treemap": grouped["sector_treemap"],
			"variance_bar": grouped["variance_bar"],
		},
		"status": "ready",
	}
	return {
		"dashboard_state": dashboard_state,
		"status": "success",
	}


def explain_variance_node(state: WorkflowState) -> WorkflowState:
	"""Explain the currently selected variance driver."""
	root_cause = str(state.get("filters", {}).get("selected_root_cause", ""))
	variance_bar = state["grouped"]["variance_bar"]
	selected_row = variance_bar.loc[variance_bar["Root_Cause"].eq(root_cause)]
	variance_total = float(selected_row["Variance_vs_Budget"].iloc[0]) if not selected_row.empty else 0.0
	explainer = ClaudeVarianceExplainer(api_key=state.get("config", {}).get("anthropic_api_key"))
	explanation = explainer.explain_variance(
		{
			"root_cause": root_cause,
			"variance_total": variance_total,
			"variance_drift": state["grouped"]["variance_drift"],
			"concentration_risk": state["grouped"]["concentration_risk"],
			"filters": state.get("filters", _default_filters()),
		}
	)
	return {"explanation": explanation, "status": "success"}


def email_risk_owner_node(state: WorkflowState) -> WorkflowState:
	"""Launch an email draft for the selected risk owner."""
	selected_risk_id = str(state.get("filters", {}).get("selected_risk_id", ""))
	risk_frame = state["tables"]["open_risks"]
	risk_record = next((row for row in risk_frame if str(row["risk_id"]) == selected_risk_id), None)
	if risk_record is None:
		return {
			"email_result": {
				"status": "error",
				"message": "No matching risk was available for the email action.",
			},
			"status": "error",
		}
	result = open_risk_owner_email(risk_record, {"summary": ", ".join(_build_summary_lines(state["metrics"], state["grouped"]))})
	return {"email_result": result, "status": result["status"]}


def export_report_node(state: WorkflowState) -> WorkflowState:
	"""Export a PDF summary for the current filtered dashboard state."""
	config = load_config(Path(state.get("config", {}).get("project_root")) if state.get("config") else None)
	result = export_dashboard_pdf(
		export_dir=config.export_dir,
		metrics=state["metrics"],
		filters=state.get("filters", _default_filters()),
		summary_lines=_build_summary_lines(state["metrics"], state["grouped"]),
	)
	return {"export_result": result, "status": result["status"]}


def send_email_report_node(state: WorkflowState) -> WorkflowState:
	"""Generate and dispatch today's analytics report email."""
	config = load_config(Path(state.get("config", {}).get("project_root")) if state.get("config") else None)
	metrics = state["metrics"]
	grouped = state["grouped"]
	filters = state.get("filters", _default_filters())
	body = format_daily_report_body(
		metrics=metrics,
		grouped=grouped,
		analysis_date=config.analysis_date,
		filters=filters,
	)
	subject = f"Daily Analytics Report — {config.analysis_date.isoformat()}"
	result = email_tool.send_report(
		to="analytics-team@example.com",
		subject=subject,
		body=body,
	)
	return {
		"email_report_result": result,
		"daily_report_summary": body,
		"status": result.get("status", "success"),
	}


def apply_dashboard_filters(
	spend_header: pd.DataFrame,
	spend_detail: pd.DataFrame,
	risk: pd.DataFrame,
	filters: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Apply current filter selections across all datasets."""
	filtered_header = spend_header.copy()
	filtered_risk = risk.copy()

	sector = filters.get("sector", "All")
	po_status = filters.get("po_status", "All")
	addressable = filters.get("addressable", "All")
	risk_status = filters.get("risk_status", "All")
	time_range = str(filters.get("time_range", "month")).lower()

	if sector != "All":
		filtered_header = filtered_header.loc[filtered_header["Sector"].eq(sector)]
	if po_status != "All":
		filtered_header = filtered_header.loc[filtered_header["PO_Status"].eq(po_status)]
	if addressable != "All":
		filtered_header = filtered_header.loc[filtered_header["Addressable_Flag"].eq(addressable)]
	if risk_status != "All":
		filtered_risk = filtered_risk.loc[filtered_risk["Risk_Status"].eq(risk_status)]

	anchor_date = _resolve_anchor_date(filtered_header)
	if anchor_date is not None:
		start_date, end_date = _time_range_bounds(anchor_date, time_range)
		filtered_header = filtered_header.loc[
			filtered_header["Last_Updated_Timestamp"].dt.date.between(start_date, end_date)
		]
		filtered_risk = _filter_risk_by_time_range(filtered_risk, start_date, end_date)

	allowed_po_numbers = filtered_header["PO_Number"].unique().tolist()
	filtered_detail = spend_detail.loc[spend_detail["PO_Number"].isin(allowed_po_numbers)].copy()
	if sector != "All":
		filtered_detail = filtered_detail.loc[filtered_detail["Sector"].eq(sector)]
	if anchor_date is not None:
		start_date, end_date = _time_range_bounds(anchor_date, time_range)
		filtered_detail = filtered_detail.loc[
			filtered_detail["Last_Updated_Timestamp"].dt.date.between(start_date, end_date)
		]

	return filtered_header, filtered_detail, filtered_risk


def _normalize_spend_header(frame: pd.DataFrame) -> pd.DataFrame:
	normalized = frame.rename(columns={"Business_Sector": "Sector"}).copy()
	normalized["Last_Updated_Timestamp"] = pd.to_datetime(
		normalized["Last_Updated_Timestamp"],
		errors="coerce",
	)
	normalized["Addressable_Flag"] = normalized["Addressable_Flag"].astype(str)
	normalized["PO_Status"] = normalized["PO_Status"].astype(str)
	return normalized


def _normalize_spend_detail(frame: pd.DataFrame) -> pd.DataFrame:
	normalized = frame.rename(columns={"Root_Cause_Code": "Root_Cause"}).copy()
	normalized["Last_Updated_Timestamp"] = pd.to_datetime(
		normalized["Last_Updated_Timestamp"],
		errors="coerce",
	)
	normalized["Sector"] = normalized["Sector"].astype(str)
	normalized["Root_Cause"] = normalized["Root_Cause"].astype(str)
	return normalized


def _normalize_risk(frame: pd.DataFrame, analysis_date: date) -> pd.DataFrame:
	normalized = frame.rename(
		columns={
			"Risk #": "Risk_ID",
			"Risk Description": "Risk_Description",
			"Risk Owner": "Risk_Owner",
			"Risk Status": "Risk_Status",
			"Risk Category": "Risk_Category",
			"Risk Level": "Risk_Level",
			"Risk ERM Type": "Risk_ERM_Type",
			"Open Date": "Open_Date",
			"Closed Date": "Closed_Date",
			"Days Open": "Days_Open",
		}
	).copy()
	normalized["Open_Date"] = pd.to_datetime(normalized["Open_Date"], errors="coerce")
	normalized["Closed_Date"] = pd.to_datetime(normalized["Closed_Date"], errors="coerce")
	normalized["Risk_Status"] = normalized["Risk_Status"].astype(str)
	normalized["Risk_Level"] = normalized["Risk_Level"].astype(str)
	normalized["Active"] = normalized["Risk_Status"].str.lower().ne("closed") | normalized["Closed_Date"].isna()

	derived_days_open = (normalized["Closed_Date"].fillna(pd.Timestamp(analysis_date)) - normalized["Open_Date"]).dt.days
	normalized["Days_Open"] = normalized["Days_Open"].fillna(derived_days_open).fillna(0).astype(int)
	normalized["Impact_Score"] = normalized["Risk_Level"].map(IMPACT_SCORE_MAP).fillna(1).astype(int)
	normalized["Likelihood_Score"] = pd.cut(
		normalized["Days_Open"],
		bins=[-1, 30, 90, 180, 365, float("inf")],
		labels=[1, 2, 3, 4, 5],
	).astype(int)
	return normalized


def _build_filter_options(spend_header: pd.DataFrame, risk: pd.DataFrame) -> dict[str, list[str]]:
	return {
		"sectors": ["All", *sorted(spend_header["Sector"].dropna().astype(str).unique().tolist())],
		"po_statuses": ["All", *sorted(spend_header["PO_Status"].dropna().astype(str).unique().tolist())],
		"addressable_options": ["All", *sorted(spend_header["Addressable_Flag"].dropna().astype(str).unique().tolist())],
		"risk_statuses": ["All", *sorted(risk["Risk_Status"].dropna().astype(str).unique().tolist())],
	}


def _group_sector_spend(frame: pd.DataFrame) -> pd.DataFrame:
	return (
		frame.groupby(["Sector", "PO_Status"], as_index=False)["PO_Total_Amount"]
		.sum()
		.sort_values("PO_Total_Amount", ascending=False)
	)


def _group_root_cause_variance(frame: pd.DataFrame) -> pd.DataFrame:
	grouped = (
		frame.groupby(["Root_Cause", "Sector"], as_index=False)["Variance_vs_Budget"]
		.sum()
		.sort_values("Variance_vs_Budget", key=lambda series: series.abs(), ascending=False)
	)
	return grouped


def _build_trend_frame(spend_header: pd.DataFrame, spend_detail: pd.DataFrame) -> pd.DataFrame:
	spend_monthly = (
		spend_header.set_index("Last_Updated_Timestamp")["PO_Total_Amount"].resample("ME").sum().rename("Monthly_Spend")
	)
	variance_monthly = (
		spend_detail.set_index("Last_Updated_Timestamp")["Variance_vs_Budget"].resample("ME").sum().rename("Monthly_Variance")
	)
	combined = pd.concat([spend_monthly, variance_monthly], axis=1).fillna(0.0).reset_index()
	combined["Cumulative_Variance"] = combined["Monthly_Variance"].cumsum()
	return combined


def _group_risk_heatmap(risk: pd.DataFrame) -> pd.DataFrame:
	grouped = risk.groupby(["Likelihood_Score", "Impact_Score"], as_index=False)["Risk_ID"].count()
	grouped = grouped.rename(columns={"Risk_ID": "Risk_Count"})
	full_grid = pd.MultiIndex.from_product([range(1, 6), range(1, 6)], names=["Likelihood_Score", "Impact_Score"]).to_frame(index=False)
	return full_grid.merge(grouped, on=["Likelihood_Score", "Impact_Score"], how="left").fillna({"Risk_Count": 0})


def _build_open_risk_table(risk: pd.DataFrame) -> list[dict[str, Any]]:
	open_risks = risk.loc[risk["Active"]].sort_values("Days_Open", ascending=False).head(12)
	return [
		{
			"risk_id": str(row["Risk_ID"]),
			"risk_description": str(row["Risk_Description"]),
			"risk_owner": str(row["Risk_Owner"]),
			"risk_status": str(row["Risk_Status"]),
			"days_open": int(row["Days_Open"]),
		}
		for _, row in open_risks.iterrows()
	]


def _compute_variance_drift(spend_detail: pd.DataFrame) -> dict[str, Any]:
	monthly = spend_detail.set_index("Last_Updated_Timestamp")["Variance_vs_Budget"].resample("ME").sum()
	rolling = monthly.rolling(3, min_periods=2).mean().dropna()
	if len(rolling) < 2:
		return {"direction": "stable", "delta": 0.0}
	delta = float(rolling.iloc[-1] - rolling.iloc[-2])
	direction = "up" if delta > 0 else "down" if delta < 0 else "stable"
	return {"direction": direction, "delta": round(delta, 2)}


def _compute_concentration_risk(spend_header: pd.DataFrame) -> dict[str, Any]:
	sector_spend = spend_header.groupby("Sector")["PO_Total_Amount"].sum().sort_values(ascending=False)
	if sector_spend.empty:
		return {"top_share": 0.0, "is_concentrated": False}
	top_count = max(1, int(len(sector_spend) * 0.2))
	top_share = float(sector_spend.head(top_count).sum() / sector_spend.sum())
	return {
		"top_share": round(top_share, 4),
		"is_concentrated": top_share >= 0.8,
	}


def _build_summary_lines(metrics: dict[str, Any], grouped: dict[str, Any]) -> list[str]:
	drift = grouped["variance_drift"]
	concentration = grouped["concentration_risk"]
	return [
		f"Total PO volume in the current view is {metrics['total_po_volume']}.",
		f"Average variance is ${metrics['average_variance']:,.2f} and addressable spend share is {metrics['addressable_spend_pct']:.2f}%.",
		f"Active risk count is {metrics['active_risk_count']}.",
		f"Variance drift is currently {drift['direction']} with a delta of ${drift['delta']:,.2f}.",
		f"Top-spend concentration share is {concentration['top_share'] * 100:.1f}% of filtered spend.",
	]


def _serialize_config(config: DashboardConfig) -> dict[str, Any]:
	serialized = asdict(config)
	return {key: str(value) if isinstance(value, Path) else value for key, value in serialized.items()}


def _default_filters() -> dict[str, str]:
	return {
		"sector": "All",
		"po_status": "All",
		"addressable": "All",
		"risk_status": "All",
		"time_range": "month",
	}


def _resolve_anchor_date(spend_header: pd.DataFrame) -> date | None:
	if spend_header.empty or spend_header["Last_Updated_Timestamp"].dropna().empty:
		return None
	return spend_header["Last_Updated_Timestamp"].max().date()


def _time_range_bounds(anchor_date: date, time_range: str) -> tuple[date, date]:
	normalized = time_range.lower()
	if normalized == "today":
		return anchor_date, anchor_date
	if normalized == "week":
		start_of_week = anchor_date - timedelta(days=anchor_date.weekday())
		return start_of_week, anchor_date
	# default = month
	start_of_month = anchor_date.replace(day=1)
	return start_of_month, anchor_date


def _filter_risk_by_time_range(risk: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
	if risk.empty:
		return risk
	open_dates = risk["Open_Date"].dt.date
	close_dates = risk["Closed_Date"].dt.date
	close_dates_filled = close_dates.fillna(end_date)
	# Keep risks that overlap the selected range window.
	mask = (open_dates <= end_date) & (close_dates_filled >= start_date)
	return risk.loc[mask]


def _validate_columns(frame: pd.DataFrame, required: set[str], dataset_name: str) -> None:
	missing = sorted(required.difference(frame.columns))
	if missing:
		raise ValueError(f"{dataset_name} is missing required columns: {', '.join(missing)}")
