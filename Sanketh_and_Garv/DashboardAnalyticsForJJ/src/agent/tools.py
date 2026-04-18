"""Side-effect tools for email launch and PDF export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
import webbrowser

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def open_risk_owner_email(risk_record: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
	"""Open a mailto link for the selected risk owner."""
	risk_id = str(risk_record.get("risk_id") or "")
	risk_owner = str(risk_record.get("risk_owner") or "")
	recipient_email = _resolve_owner_email(risk_owner)
	description = str(risk_record.get("risk_description") or "")
	days_open = str(risk_record.get("days_open") or "")
	if not risk_id or not risk_owner:
		return {
			"status": "error",
			"message": "A risk identifier and risk owner are required before launching email.",
		}

	subject = f"Status update requested for Risk ID {risk_id}"
	body = (
		f"Recipient: {recipient_email}\n"
		f"Risk Owner: {risk_owner}\n"
		f"Risk ID: {risk_id}\n"
		f"Description: {description}\n"
		f"Status: {risk_record.get('risk_status', 'Unknown')}\n"
		f"Days Open: {days_open}\n"
		f"Current dashboard context: {context.get('summary', 'Filtered dashboard view')}\n\n"
		"Please provide a status update, any mitigation progress, and the expected next checkpoint."
	)
	mailto_url = f"mailto:{quote(recipient_email)}?subject={quote(subject)}&body={quote(body)}"
	webbrowser.open(mailto_url)
	return {
		"status": "success",
		"message": f"Email draft prepared for {risk_owner}.",
		"subject": subject,
		"recipient": recipient_email,
	}


def export_dashboard_pdf(
	export_dir: Path,
	metrics: dict[str, Any],
	filters: dict[str, str],
	summary_lines: list[str],
) -> dict[str, Any]:
	"""Export a PDF summary of the current dashboard state."""
	export_dir.mkdir(parents=True, exist_ok=True)
	filename = f"dashboard_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
	export_path = export_dir / filename

	pdf = canvas.Canvas(str(export_path), pagesize=letter)
	width, height = letter
	y_position = height - 72

	pdf.setFont("Helvetica-Bold", 18)
	pdf.drawString(72, y_position, "DashboardAnalyticsForJJ Summary")
	y_position -= 28

	pdf.setFont("Helvetica", 11)
	pdf.drawString(72, y_position, f"Exported: {datetime.now().isoformat(timespec='seconds')}")
	y_position -= 24

	pdf.setFont("Helvetica-Bold", 13)
	pdf.drawString(72, y_position, "Active Filters")
	y_position -= 18
	pdf.setFont("Helvetica", 11)
	for key, value in filters.items():
		pdf.drawString(84, y_position, f"{key.replace('_', ' ').title()}: {value}")
		y_position -= 16

	y_position -= 8
	pdf.setFont("Helvetica-Bold", 13)
	pdf.drawString(72, y_position, "Executive Metrics")
	y_position -= 18
	pdf.setFont("Helvetica", 11)
	for key, value in metrics.items():
		pdf.drawString(84, y_position, f"{key.replace('_', ' ').title()}: {value}")
		y_position -= 16

	y_position -= 8
	pdf.setFont("Helvetica-Bold", 13)
	pdf.drawString(72, y_position, "Narrative Summary")
	y_position -= 18
	pdf.setFont("Helvetica", 11)
	for line in summary_lines:
		if y_position < 72:
			pdf.showPage()
			pdf.setFont("Helvetica", 11)
			y_position = height - 72
		pdf.drawString(84, y_position, line)
		y_position -= 16

	pdf.save()
	return {
		"status": "success",
		"message": f"PDF summary saved to {export_path}",
		"path": str(export_path),
		"filename": filename,
	}


def _resolve_owner_email(risk_owner: str) -> str:
	normalized = "".join(ch.lower() if ch.isalnum() else "." for ch in risk_owner).strip(".")
	normalized = ".".join(part for part in normalized.split(".") if part)
	return f"{normalized or 'risk.owner'}@dashboardanalytics.local"
