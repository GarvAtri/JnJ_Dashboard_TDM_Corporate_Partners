"""Email reporting tool for daily analytics dispatch."""

from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import quote
import webbrowser


class _EmailTool:
	"""Mailto-based email dispatch helper."""

	def send_report(self, to: str, subject: str, body: str) -> dict[str, str]:
		"""Construct and open a mailto URL for a report email."""
		mailto_url = f"mailto:{quote(to)}?subject={quote(subject)}&body={quote(body)}"
		webbrowser.open(mailto_url)
		return {
			"status": "success",
			"mailto_url": mailto_url,
			"to": to,
			"subject": subject,
		}


email_tool = _EmailTool()


def format_daily_report_body(
	metrics: dict[str, Any],
	grouped: dict[str, Any],
	analysis_date: date,
	filters: dict[str, str],
) -> str:
	"""Build a concise structured daily analytics summary for email body."""
	variance_bar = grouped.get("variance_bar")
	top_root = "N/A"
	top_root_value = 0.0
	if variance_bar is not None and len(variance_bar):
		top = variance_bar.iloc[0]
		top_root = str(top.get("Root_Cause", "N/A"))
		top_root_value = float(top.get("Variance_vs_Budget", 0.0))

	risk_hist = grouped.get("aging_histogram")
	long_tail_risk_count = 0
	if risk_hist is not None and len(risk_hist):
		long_tail_risk_count = int((risk_hist["Days_Open"] > 365).sum())

	filter_line = ", ".join(f"{key}: {value}" for key, value in filters.items())
	body_lines = [
		f"Daily Analytics Report - {analysis_date.isoformat()}",
		"",
		"Executive Summary",
		f"- Total PO Volume: {metrics.get('total_po_volume', 0)}",
		f"- Average Variance: ${float(metrics.get('average_variance', 0.0)):,.2f}",
		f"- Active Risk Count: {metrics.get('active_risk_count', 0)}",
		f"- Addressable Spend %: {float(metrics.get('addressable_spend_pct', 0.0)):.2f}%",
		"",
		"Spend and Variance Insights",
		f"- Top variance driver: {top_root} (${top_root_value:,.0f}).",
		f"- Total spend in current context: ${float(metrics.get('total_spend', 0.0)):,.0f}.",
		"",
		"Risk and Governance Insights",
		f"- Active risks currently tracked: {metrics.get('active_risk_count', 0)}.",
		f"- Long-tail risks open more than 365 days: {long_tail_risk_count}.",
		"",
		f"Filter Context: {filter_line}",
	]
	return "\n".join(body_lines)
