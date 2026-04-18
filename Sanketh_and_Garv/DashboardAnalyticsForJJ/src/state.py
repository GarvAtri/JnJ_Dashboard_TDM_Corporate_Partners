"""Reflex state for DashboardAnalyticsForJJ."""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import reflex as rx
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from src.agent.graph import (
	run_chat_workflow,
	run_dashboard_workflow,
	run_email_workflow,
	run_export_workflow,
	run_send_email_report_workflow,
	run_save_response_workflow,
	run_variance_explanation_workflow,
)
from src.agent.file_scraper import scrape_all_files
from src.agent.column_analyzer import ClaudeColumnAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".json", ".pdf"}
UPLOAD_DIR = PROJECT_ROOT / "uploads"


class DashboardState(rx.State):
	"""Reactive state container for the analytics dashboard."""
	
	active_mode: str = "overview"
	nav_collapsed: bool = False
	chat_panel_mode: str = "left"
	chat_panel_collapsed: bool = False
	chat_float_top: int = 110
	chat_float_left: int = 980
	expanded_chart_id: str = ""

	selected_sector: str = "All"
	selected_po_status: str = "All"
	selected_addressable: str = "All"
	selected_risk_status: str = "All"
	selected_time_range: str = "month"
	show_filters: bool = True
	show_advanced_filters: bool = False

	sector_options: list[str] = ["All"]
	po_status_options: list[str] = ["All"]
	addressable_options: list[str] = ["All"]
	risk_status_options: list[str] = ["All"]
	time_range_options: list[str] = ["today", "week", "month"]

	total_po_volume_display: str = "0"
	average_variance_display: str = "$0.00"
	active_risk_count_display: str = "0"
	addressable_spend_pct_display: str = "0.00%"

	sector_treemap_figure: Figure = go.Figure()
	root_cause_variance_figure: Figure = go.Figure()
	trend_and_seasonality_figure: Figure = go.Figure()
	risk_heatmap_figure: Figure = go.Figure()
	aging_risk_histogram_figure: Figure = go.Figure()

	open_risks: list[dict] = []
	sector_breakdown: list[dict] = []
	variance_breakdown: list[dict] = []
	dashboard_insights: list[str] = []
	variance_explanation: str = "Hover over a variance bar to let Claude explain the selected root cause."
	last_action_message: str = ""
	load_error: str = ""
	is_loading: bool = False

	chat_messages: list[dict[str, Any]] = [
		{
			"id": "assistant-welcome",
			"role": "assistant",
			"content": "Welcome to the Claude chatbot panel. Ask about spend trends, root causes, risk clusters, or chart interpretation.",
			"saved": False,
			"timestamp": "",
		}
	]
	chat_input: str = ""
	is_chat_loading: bool = False
	chat_error: str = ""
	chat_memory_preview: str = ""
	chat_scroll_counter: int = 0

	def _current_filters(self) -> dict[str, str]:
		return {
			"sector": self.selected_sector,
			"po_status": self.selected_po_status,
			"addressable": self.selected_addressable,
			"risk_status": self.selected_risk_status,
			"time_range": self.selected_time_range,
		}

	@staticmethod
	def _ui_state_file() -> Path:
		return PROJECT_ROOT / ".claude" / "memory" / "ui_state.json"

	@classmethod
	def _read_persisted_ui_state(cls) -> dict[str, Any]:
		state_file = cls._ui_state_file()
		if not state_file.exists():
			return {
				"nav_collapsed": False,
				"chat_panel_mode": "left",
				"chat_panel_collapsed": False,
				"chat_float_top": 110,
				"chat_float_left": 980,
				"selected_time_range": "month",
				"show_filters": True,
				"show_advanced_filters": False,
			}
		try:
			payload = json.loads(state_file.read_text(encoding="utf-8"))
			return {
				"nav_collapsed": bool(payload.get("nav_collapsed", False)),
				"chat_panel_mode": str(payload.get("chat_panel_mode", "left")),
				"chat_panel_collapsed": bool(payload.get("chat_panel_collapsed", False)),
				"chat_float_top": int(payload.get("chat_float_top", 110)),
				"chat_float_left": int(payload.get("chat_float_left", 980)),
				"selected_time_range": str(payload.get("selected_time_range", "month")),
				"show_filters": bool(payload.get("show_filters", True)),
				"show_advanced_filters": bool(payload.get("show_advanced_filters", False)),
			}
		except (json.JSONDecodeError, OSError):
			return {
				"nav_collapsed": False,
				"chat_panel_mode": "left",
				"chat_panel_collapsed": False,
				"chat_float_top": 110,
				"chat_float_left": 980,
				"selected_time_range": "month",
				"show_filters": True,
				"show_advanced_filters": False,
			}

	@classmethod
	def _write_persisted_ui_state(cls, payload: dict[str, Any]) -> None:
		state_file = cls._ui_state_file()
		state_file.parent.mkdir(parents=True, exist_ok=True)
		state_file.write_text(json.dumps(payload), encoding="utf-8")

	def _persist_ui_state(self) -> None:
		self._write_persisted_ui_state(
			{
				"nav_collapsed": self.nav_collapsed,
				"chat_panel_mode": self.chat_panel_mode,
				"chat_panel_collapsed": self.chat_panel_collapsed,
				"chat_float_top": self.chat_float_top,
				"chat_float_left": self.chat_float_left,
				"selected_time_range": self.selected_time_range,
				"show_filters": self.show_filters,
				"show_advanced_filters": self.show_advanced_filters,
			}
		)

	@rx.event(background=True)
	async def load_dashboard(self) -> None:
		"""Load and refresh the dashboard asynchronously."""
		persisted_ui_state = self._read_persisted_ui_state()
		async with self:
			self.is_loading = True
			self.load_error = ""
			self.nav_collapsed = bool(persisted_ui_state.get("nav_collapsed", False))
			self.chat_panel_mode = str(persisted_ui_state.get("chat_panel_mode", "left"))
			self.chat_panel_collapsed = bool(persisted_ui_state.get("chat_panel_collapsed", False))
			self.chat_float_top = int(persisted_ui_state.get("chat_float_top", 110))
			self.chat_float_left = int(persisted_ui_state.get("chat_float_left", 980))
			self.selected_time_range = str(persisted_ui_state.get("selected_time_range", "month"))
			self.show_filters = bool(persisted_ui_state.get("show_filters", True))
			self.show_advanced_filters = bool(persisted_ui_state.get("show_advanced_filters", False))
			filters = self._current_filters()
		result = await asyncio.to_thread(run_dashboard_workflow, filters, PROJECT_ROOT)
		async with self:
			self.is_loading = False
			if result.get("status") != "success":
				self.load_error = "; ".join(result.get("errors", ["Dashboard load failed."]))
				return
			dashboard_state = result["dashboard_state"]
			metrics = dashboard_state["metrics"]
			filter_options = dashboard_state["filter_options"]
			self.sector_options = filter_options.get("sectors", ["All"])
			self.po_status_options = filter_options.get("po_statuses", ["All"])
			self.addressable_options = filter_options.get("addressable_options", ["All"])
			self.risk_status_options = filter_options.get("risk_statuses", ["All"])
			self.total_po_volume_display = f"{metrics.get('total_po_volume', 0):,}"
			self.average_variance_display = f"${metrics.get('average_variance', 0.0):,.2f}"
			self.active_risk_count_display = f"{metrics.get('active_risk_count', 0):,}"
			self.addressable_spend_pct_display = f"{metrics.get('addressable_spend_pct', 0.0):.2f}%"
			self.sector_treemap_figure = go.Figure(dashboard_state["charts"].get("sector_treemap", {}))
			self.root_cause_variance_figure = go.Figure(dashboard_state["charts"].get("root_cause_variance", {}))
			self.trend_and_seasonality_figure = go.Figure(dashboard_state["charts"].get("trend_and_seasonality", {}))
			self.risk_heatmap_figure = go.Figure(dashboard_state["charts"].get("risk_heatmap", {}))
			self.aging_risk_histogram_figure = go.Figure(dashboard_state["charts"].get("aging_risk_histogram", {}))
			self.open_risks = dashboard_state["tables"].get("open_risks", [])
			# Store data for rich chat context
			raw_sector = dashboard_state.get("grouped", {}).get("sector_treemap")
			if raw_sector is not None and hasattr(raw_sector, "to_dict"):
				self.sector_breakdown = [
					{"sector": str(r.get("Sector", "")), "po_status": str(r.get("PO_Status", "")), "amount": float(r.get("PO_Total_Amount", 0))}
					for r in raw_sector.head(15).to_dict("records")
				]
			raw_variance = dashboard_state.get("grouped", {}).get("variance_bar")
			if raw_variance is not None and hasattr(raw_variance, "to_dict"):
				self.variance_breakdown = [
					{"root_cause": str(r.get("Root_Cause", "")), "sector": str(r.get("Sector", "")), "variance": float(r.get("Variance_vs_Budget", 0))}
					for r in raw_variance.head(15).to_dict("records")
				]
			insights = dashboard_state.get("insights", [])
			self.dashboard_insights = insights
			self.variance_explanation = "\n".join(insights[:2]) if insights else self.variance_explanation

	def set_sector(self, value: str):
		self.selected_sector = value
		return DashboardState.load_dashboard

	def set_po_status(self, value: str):
		self.selected_po_status = value
		return DashboardState.load_dashboard

	def set_addressable(self, value: str):
		self.selected_addressable = value
		return DashboardState.load_dashboard

	def set_risk_status(self, value: str):
		self.selected_risk_status = value
		return DashboardState.load_dashboard

	def set_time_range(self, value: str):
		self.selected_time_range = value
		self._persist_ui_state()
		return DashboardState.load_dashboard

	def toggle_filters(self):
		self.show_filters = not self.show_filters
		self._persist_ui_state()

	def toggle_advanced_filters(self):
		self.show_advanced_filters = not self.show_advanced_filters
		self._persist_ui_state()

	@rx.var
	def time_range_label(self) -> str:
		if self.selected_time_range == "today":
			return "Today"
		if self.selected_time_range == "week":
			return "This Week"
		return "This Month"

	@rx.var
	def active_filter_chips(self) -> list[str]:
		chips = [f"Range: {self.time_range_label}"]
		if self.selected_sector != "All":
			chips.append(f"Sector: {self.selected_sector}")
		if self.selected_po_status != "All":
			chips.append(f"PO: {self.selected_po_status}")
		if self.selected_addressable != "All":
			chips.append(f"Addressable: {self.selected_addressable}")
		if self.selected_risk_status != "All":
			chips.append(f"Risk: {self.selected_risk_status}")
		return chips

	def set_chat_input(self, value: str):
		self.chat_input = value

	def handle_chat_submit(self, form_data: dict):
		"""Handle form submission from Enter key press."""
		self.chat_input = form_data.get("chat_input", "")
		return DashboardState.ask_claude

	def set_active_mode(self, mode: str):
		self.active_mode = mode

	def toggle_nav(self):
		self.nav_collapsed = not self.nav_collapsed
		self._persist_ui_state()

	def toggle_chat_panel(self):
		self.chat_panel_collapsed = not self.chat_panel_collapsed
		self._persist_ui_state()

	def move_chat_to_right(self):
		self.chat_panel_mode = "right"
		self.chat_panel_collapsed = False
		self._persist_ui_state()

	def move_chat_to_left(self):
		self.chat_panel_mode = "left"
		self.chat_panel_collapsed = False
		self._persist_ui_state()

	def set_chat_floating(self):
		self.chat_panel_mode = "floating"
		self.chat_panel_collapsed = False
		self._persist_ui_state()

	def nudge_chat(self, delta_x: int, delta_y: int):
		self.chat_float_left = max(40, min(1500, self.chat_float_left + delta_x))
		self.chat_float_top = max(20, min(860, self.chat_float_top + delta_y))
		self._persist_ui_state()

	def open_chart_modal(self, chart_id: str):
		self.expanded_chart_id = chart_id

	def close_chart_modal(self):
		self.expanded_chart_id = ""

	@rx.event(background=True)
	async def explain_variance_from_hover(self, points: list[dict[str, Any]]) -> None:
		"""Explain the hovered variance bar using the variance workflow."""
		if not points:
			return
		root_cause = str(points[0].get("y") or "")
		if not root_cause:
			return
		result = await asyncio.to_thread(
			run_variance_explanation_workflow,
			self._current_filters(),
			root_cause,
			PROJECT_ROOT,
		)
		async with self:
			explanation = result.get("explanation", {}).get("message")
			if explanation:
				self.variance_explanation = explanation

	@rx.event(background=True)
	async def email_risk_owner(self, risk_id: str) -> None:
		"""Launch the risk owner email action in the background."""
		result = await asyncio.to_thread(run_email_workflow, self._current_filters(), risk_id, PROJECT_ROOT)
		async with self:
			self.last_action_message = result.get("email_result", {}).get("message", "Email action completed.")

	@rx.event(background=True)
	async def export_report(self) -> None:
		"""Export the current dashboard view to PDF."""
		result = await asyncio.to_thread(run_export_workflow, self._current_filters(), PROJECT_ROOT)
		async with self:
			self.last_action_message = result.get("export_result", {}).get("message", "Export completed.")

	@rx.event(background=True)
	async def ask_claude(self) -> None:
		"""Send chat input to Claude through chat workflow orchestration."""
		user_message = self.chat_input.strip()
		if not user_message:
			return

		history_payload = [{"role": item["role"], "content": item["content"]} for item in self.chat_messages]
		user_message_id = f"user-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
		assistant_message_id = f"assistant-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
		user_ts = datetime.now().strftime("%I:%M %p")
		async with self:
			self.chat_error = ""
			self.is_chat_loading = True
			self.chat_messages = [
				*self.chat_messages,
				{"id": user_message_id, "role": "user", "content": user_message, "saved": False, "timestamp": user_ts},
			]
			self.chat_input = ""
			self.chat_scroll_counter += 1

		data_context = {
			"metrics": {
				"total_po_volume": self.total_po_volume_display,
				"average_variance": self.average_variance_display,
				"active_risk_count": self.active_risk_count_display,
				"addressable_spend_pct": self.addressable_spend_pct_display,
			},
			"filters": self._current_filters(),
			"open_risks": self.open_risks,
			"sector_breakdown": self.sector_breakdown,
			"variance_breakdown": self.variance_breakdown,
			"insights": self.dashboard_insights,
			"variance_explanation": self.variance_explanation,
		}

		result = await asyncio.to_thread(
			run_chat_workflow,
			user_message,
			history_payload,
			data_context,
			PROJECT_ROOT,
		)
		assistant_message = result.get("chat_state", {}).get("assistant_message", "")
		async with self:
			self.is_chat_loading = False
			if not assistant_message:
				self.chat_error = "Claude response was empty. Please try again."
				return
			reply_ts = datetime.now().strftime("%I:%M %p")
			self.chat_messages = [
				*self.chat_messages,
				{
					"id": assistant_message_id,
					"role": "assistant",
					"content": assistant_message,
					"saved": False,
					"timestamp": reply_ts,
				},
			]
			self.chat_scroll_counter += 1
			self.chat_memory_preview = result.get("chat_state", {}).get("saved_memory", "")[-1200:]

	@rx.event(background=True)
	async def save_response(self, message_id: str) -> None:
		"""Persist one assistant message in durable chat memory."""
		selected = next(
			(
				item for item in self.chat_messages
				if str(item.get("id")) == str(message_id) and item.get("role") == "assistant"
			),
			None,
		)
		if not selected:
			return
		result = await asyncio.to_thread(run_save_response_workflow, str(selected.get("content", "")), PROJECT_ROOT)
		updated_messages = []
		for item in self.chat_messages:
			if str(item.get("id")) == str(message_id):
				updated_messages.append({**item, "saved": True})
			else:
				updated_messages.append(item)
		async with self:
			self.chat_messages = updated_messages
			self.last_action_message = result.get("save_result", {}).get("message", "Response saved.")

	@rx.event(background=True)
	async def save_last_response(self) -> None:
		"""Save the most recent unsaved assistant message."""
		last_msg = None
		for msg in reversed(self.chat_messages):
			if msg.get("role") == "assistant" and not msg.get("saved"):
				last_msg = msg
				break
		if not last_msg:
			return
		result = await asyncio.to_thread(
			run_save_response_workflow, str(last_msg.get("content", "")), PROJECT_ROOT,
		)
		updated = []
		for item in self.chat_messages:
			if str(item.get("id")) == str(last_msg["id"]):
				updated.append({**item, "saved": True})
			else:
				updated.append(item)
		async with self:
			self.chat_messages = updated
			self.last_action_message = result.get("save_result", {}).get("message", "Response saved.")

	@rx.event(background=True)
	async def send_email_report(self) -> None:
		"""Send today's analytics summary through default email client."""
		result = await asyncio.to_thread(
			run_send_email_report_workflow,
			self._current_filters(),
			PROJECT_ROOT,
		)
		async with self:
			if result.get("status") == "success":
				self.last_action_message = "Daily analytics email draft opened in your default mail client."
			else:
				self.chat_error = "; ".join(result.get("errors", ["Unable to send analytics report email."]))

	# ── File Upload ──────────────────────────────────────────────

	uploaded_files: list[dict[str, str]] = []
	upload_progress: int = 0
	is_uploading: bool = False
	upload_error: str = ""
	show_upload_modal: bool = False

	def toggle_upload_modal(self):
		self.show_upload_modal = not self.show_upload_modal
		if not self.show_upload_modal:
			self.upload_error = ""

	async def handle_upload(self, files: list[rx.UploadFile]):
		"""Process uploaded files (Excel, JSON, PDF)."""
		UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
		self.is_uploading = True
		self.upload_error = ""
		self.upload_progress = 0
		yield

		total = len(files)
		succeeded: list[dict[str, str]] = []
		errors: list[str] = []

		for idx, file in enumerate(files):
			filename = file.filename or f"file_{idx}"
			ext = Path(filename).suffix.lower()

			if ext not in ALLOWED_EXTENSIONS:
				errors.append(f"{filename}: unsupported type ({ext}). Allowed: .xlsx, .xls, .json, .pdf")
				continue

			dest = UPLOAD_DIR / filename
			try:
				content = await file.read()
				dest.write_bytes(content)
				size_kb = len(content) / 1024
				succeeded.append({
					"name": filename,
					"type": ext.lstrip(".").upper(),
					"size": f"{size_kb:.1f} KB",
					"path": str(dest),
					"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
				})
			except Exception as exc:
				errors.append(f"{filename}: {exc}")

			self.upload_progress = int(((idx + 1) / total) * 100)
			yield

		self.uploaded_files = [*self.uploaded_files, *succeeded]
		self.is_uploading = False
		self.upload_progress = 100

		if errors:
			self.upload_error = "; ".join(errors)
		elif succeeded:
			self.last_action_message = f"Uploaded {len(succeeded)} file(s) successfully."
		yield

	def remove_uploaded_file(self, file_name: str):
		"""Remove a file from the uploaded list and disk."""
		target = next((f for f in self.uploaded_files if f["name"] == file_name), None)
		if target:
			path = Path(target["path"])
			if path.exists():
				path.unlink()
			self.uploaded_files = [f for f in self.uploaded_files if f["name"] != file_name]

	def clear_all_uploads(self):
		"""Remove all uploaded files from disk and state."""
		for f in self.uploaded_files:
			path = Path(f["path"])
			if path.exists():
				path.unlink()
		self.uploaded_files = []
		self.upload_error = ""

	# ── File Scraping / Insights ─────────────────────────────────

	file_insights: list[dict[str, Any]] = []
	is_scraping: bool = False
	scrape_error: str = ""
	show_insights_panel: bool = False

	def toggle_insights_panel(self):
		self.show_insights_panel = not self.show_insights_panel

	@rx.event(background=True)
	async def scrape_uploaded_files(self) -> None:
		"""Scrape all uploaded files to extract metadata and structure."""
		async with self:
			if not self.uploaded_files:
				self.scrape_error = "No files uploaded yet."
				return
			file_paths = [f["path"] for f in self.uploaded_files]
			self.is_scraping = True
			self.scrape_error = ""
			self.show_insights_panel = True

		try:
			results = await asyncio.to_thread(scrape_all_files, file_paths)
		except Exception as exc:
			async with self:
				self.is_scraping = False
				self.scrape_error = f"Scraping failed: {exc}"
			return

		async with self:
			self.is_scraping = False
			errors = [r["error"] for r in results if "error" in r]
			if errors:
				self.scrape_error = "; ".join(errors)
			self.file_insights = [r for r in results if "error" not in r]
			if self.file_insights:
				self.last_action_message = f"Scraped {len(self.file_insights)} file(s) successfully."

	def clear_file_insights(self):
		self.file_insights = []
		self.scrape_error = ""

	# ── AI Column Relevance Analysis ────────────────────────────

	column_relevance_report: dict[str, Any] = {}
	is_analyzing_columns: bool = False
	column_analysis_error: str = ""

	@rx.event(background=True)
	async def analyze_columns_with_claude(self) -> None:
		"""Use Claude to classify column relevance from scraped insights."""
		async with self:
			if not self.file_insights:
				self.column_analysis_error = "Scrape files first before analyzing."
				return
			insights_copy = list(self.file_insights)
			self.is_analyzing_columns = True
			self.column_analysis_error = ""
			self.column_relevance_report = {}

		try:
			analyzer = ClaudeColumnAnalyzer()
			report = await asyncio.to_thread(analyzer.analyze, insights_copy)
		except Exception as exc:
			async with self:
				self.is_analyzing_columns = False
				self.column_analysis_error = f"Analysis failed: {exc}"
			return

		async with self:
			self.is_analyzing_columns = False
			if "error" in report and not report.get("columns"):
				self.column_analysis_error = report["error"]
			else:
				self.column_relevance_report = report
				self.last_action_message = (
					f"Column analysis complete — {report.get('relevant_count', 0)} relevant, "
					f"{report.get('irrelevant_count', 0)} irrelevant."
				)

	def clear_column_analysis(self):
		self.column_relevance_report = {}
		self.column_analysis_error = ""
