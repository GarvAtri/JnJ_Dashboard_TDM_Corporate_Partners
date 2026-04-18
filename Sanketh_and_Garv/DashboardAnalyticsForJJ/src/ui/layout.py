"""Dashboard page layout."""

from __future__ import annotations

import reflex as rx

from src.state import DashboardState
from src.ui.chat_panel import chat_panel
from src.ui.components import (
	analytics_mode_tabs,
	chart_card,
	dashboard_controls_panel,
	expanded_chart_modal,
	file_upload_panel,
	metric_card,
	predictive_insight_card,
	risk_action_list,
	risk_owner_card,
)


DEEP_SPACE_NAVY = "#0B1221"
SOFT_GRAPHITE = "#1E2635"
FROST_WHITE = "#F5F7FA"
SLATE_GREY = "#8A93A6"
ELECTRIC_CYAN = "#3EE7E0"
AZURE_PULSE_BLUE = "#4C8DFF"


def dashboard_page() -> rx.Component:
	"""Render the complete dashboard page."""
	return rx.box(
		rx.cond(
			DashboardState.chat_panel_mode == "left",
			rx.hstack(
				_navigation_rail(),
				chat_panel(),
				_dashboard_content(),
				align="start",
				spacing="0",
				width="100%",
				height="100vh",
			),
			rx.cond(
				DashboardState.chat_panel_mode == "right",
				rx.hstack(
					_navigation_rail(),
					_dashboard_content(),
					chat_panel(),
					align="start",
					spacing="0",
					width="100%",
					height="100vh",
				),
				rx.box(
					rx.hstack(
						_navigation_rail(),
						_dashboard_content(),
						align="start",
						spacing="0",
						width="100%",
						height="100vh",
					),
					chat_panel(),
					width="100%",
					height="100vh",
				),
			),
		),
		expanded_chart_modal(),
		file_upload_panel(),
		overflow="hidden",
		min_height="100vh",
		background=DEEP_SPACE_NAVY,
	)


def _dashboard_content() -> rx.Component:
	return rx.box(
		rx.vstack(
			_title_bar(),
			_status_strip(),
			analytics_mode_tabs(),
			dashboard_controls_panel(),
			_metric_row(),
			_mode_sections(),
			spacing="3",
			width="100%",
			align="stretch",
		),
		flex="1",
		min_width="0",
		overflow_y="auto",
		height="100vh",
		padding="1rem 1.25rem",
		background=(
			"radial-gradient(circle at 10% 10%, rgba(62, 231, 224, 0.12) 0%, transparent 35%),"
			"radial-gradient(circle at 80% 0%, rgba(166, 107, 255, 0.12) 0%, transparent 28%),"
			"linear-gradient(180deg, #0B1221 0%, #0D1425 100%)"
		),
	)


def _navigation_rail() -> rx.Component:
	return rx.box(
		rx.vstack(
			rx.hstack(
				rx.button(
					rx.cond(DashboardState.nav_collapsed, ">", "<"),
					on_click=DashboardState.toggle_nav,
					title="Collapse or expand navigation",
					background="rgba(62, 231, 224, 0.14)",
					border="1px solid #3EE7E0",
					color=ELECTRIC_CYAN,
					font_size="12px",
					width="28px",
					height="28px",
					border_radius="8px",
					padding="0",
					transition="all 240ms ease-in-out",
					_hover={"background": "rgba(62, 231, 224, 0.22)", "transform": "translateY(-1px)"},
				),
				rx.spacer(),
				width="100%",
			),
			rx.box(
				rx.text(
					rx.cond(DashboardState.nav_collapsed, "DB", "Dashboard"),
					color=DEEP_SPACE_NAVY,
					font_weight="700",
					font_size=rx.cond(DashboardState.nav_collapsed, "12px", "13px"),
				),
				width=rx.cond(DashboardState.nav_collapsed, "40px", "100%"),
				height="40px",
				border_radius="12px",
				background=ELECTRIC_CYAN,
				display="flex",
				align_items="center",
				justify_content="center",
			),
			rx.cond(
				DashboardState.nav_collapsed,
				rx.box(),
				rx.text("CORE", font_size="10px", letter_spacing="0.08em", color=SLATE_GREY, padding_left="0.2rem"),
			),
			rx.vstack(
				_nav_item("*", "Overview", "overview"),
				_nav_item("$", "Spend", "spend"),
				_nav_item("!", "Risk", "risk"),
				_nav_item("^", "Forecast", "forecast"),
				spacing="2",
				width="100%",
				align="stretch",
			),
			rx.box(width="100%", height="1px", background="#1F2C44"),
			rx.cond(
				DashboardState.nav_collapsed,
				rx.box(),
				rx.text("SECTIONS", font_size="10px", letter_spacing="0.08em", color=SLATE_GREY, padding_left="0.2rem"),
			),
			spacing="4",
			align="stretch",
			width="100%",
		),
		width=rx.cond(DashboardState.nav_collapsed, "70px", "260px"),
		min_width=rx.cond(DashboardState.nav_collapsed, "70px", "260px"),
		max_width=rx.cond(DashboardState.nav_collapsed, "70px", "260px"),
		height="100vh",
		padding=rx.cond(DashboardState.nav_collapsed, "0.9rem 0.4rem", "0.9rem 0.8rem"),
		background="#080F1D",
		border_right="1px solid #1A253A",
		transition="all 240ms ease-in-out",
		overflow="hidden",
	)


def _nav_item(icon_text: str, label: str, mode: str) -> rx.Component:
	is_active = DashboardState.active_mode == mode
	return rx.button(
		rx.hstack(
			rx.box(
				rx.text(
					icon_text,
					font_size="11px",
					font_weight="700",
					color=rx.cond(is_active, DEEP_SPACE_NAVY, ELECTRIC_CYAN),
				),
				background=rx.cond(is_active, ELECTRIC_CYAN, "rgba(62, 231, 224, 0.12)"),
				border=rx.cond(is_active, "1px solid #3EE7E0", "1px solid #2A3650"),
				border_radius="8px",
				padding="0.2rem 0.35rem",
				min_width="30px",
				display="flex",
				align_items="center",
				justify_content="center",
			),
			rx.cond(
				DashboardState.nav_collapsed,
				rx.box(),
				rx.text(
					label,
					font_size="12px",
					color=rx.cond(is_active, ELECTRIC_CYAN, FROST_WHITE),
					font_family="Inter, SF Pro, Poppins, sans-serif",
				),
			),
			align="center",
			spacing="2",
			justify=rx.cond(DashboardState.nav_collapsed, "center", "start"),
			width="100%",
		),
		on_click=DashboardState.set_active_mode(mode),
		title=label,
		background=rx.cond(is_active, "rgba(62, 231, 224, 0.18)", "transparent"),
		border=rx.cond(is_active, "1px solid #3EE7E0", "1px solid transparent"),
		color=rx.cond(is_active, ELECTRIC_CYAN, SLATE_GREY),
		font_size="11px",
		padding="0.45rem",
		border_radius="10px",
		font_family="Inter, SF Pro, Poppins, sans-serif",
		width="100%",
		transition="all 240ms ease-in-out",
		_hover={"color": ELECTRIC_CYAN, "border_color": "#3EE7E0"},
	)


def _title_bar() -> rx.Component:
	return rx.hstack(
		rx.heading(
			"Dashboard Analytics",
			size="5",
			color=FROST_WHITE,
			font_weight="700",
			font_family="Inter, SF Pro, Poppins, sans-serif",
			white_space="nowrap",
		),
		rx.text(
			DashboardState.time_range_label,
			color=SLATE_GREY,
			font_size="12px",
			font_family="Inter, SF Pro, Poppins, sans-serif",
		),
		rx.spacer(),
		rx.button(
			rx.hstack(
				rx.icon("upload", size=14),
				rx.text("Upload Files", font_size="12px"),
				spacing="2",
				align="center",
			),
			on_click=DashboardState.toggle_upload_modal,
			background="rgba(62, 231, 224, 0.14)",
			border="1px solid #3EE7E0",
			color=ELECTRIC_CYAN,
			font_family="Inter, SF Pro, Poppins, sans-serif",
			font_weight="500",
			padding="6px 14px",
			border_radius="8px",
			cursor="pointer",
			transition="all 240ms ease-in-out",
			_hover={"background": "rgba(62, 231, 224, 0.22)", "transform": "translateY(-1px)"},
		),
		align="center",
		spacing="3",
		width="100%",
		padding_bottom="0.2rem",
		border_bottom=f"1px solid {SOFT_GRAPHITE}",
	)


def _status_strip() -> rx.Component:
	return rx.vstack(
		rx.cond(
			DashboardState.is_loading,
			rx.box(
				rx.text("Refreshing dashboard data...", color="#4C8DFF", weight="bold"),
				background="rgba(76, 141, 255, 0.12)",
				padding="0.85rem",
				border_radius="12px",
				border="1px solid #4C8DFF",
				width="100%",
			),
			rx.box(),
		),
		rx.cond(
			DashboardState.load_error != "",
			rx.box(
				rx.text(DashboardState.load_error, color="#FF5A5F", weight="bold"),
				background="rgba(255, 90, 95, 0.12)",
				padding="0.85rem",
				border_radius="12px",
				border="1px solid #FF5A5F",
				width="100%",
			),
			rx.box(),
		),
		width="100%",
		spacing="2",
	)


def _metric_row() -> rx.Component:
	return rx.box(
		metric_card("Total PO Volume", DashboardState.total_po_volume_display, AZURE_PULSE_BLUE, "PO", "+2.1%"),
		metric_card("Avg Variance", DashboardState.average_variance_display, "#A66BFF", "VAR", "-0.8%"),
		metric_card("Active Risks", DashboardState.active_risk_count_display, ELECTRIC_CYAN, "RISK", "-1.6%"),
		metric_card("Addressable %", DashboardState.addressable_spend_pct_display, "#4CD964", "ADDR", "+0.9%"),
		display="grid",
		grid_template_columns="repeat(4, 1fr)",
		gap="12px",
		width="100%",
	)


def _mode_sections() -> rx.Component:
	return rx.vstack(
		rx.cond(DashboardState.active_mode == "overview", _overview_section(), rx.box()),
		rx.cond(DashboardState.active_mode == "spend", _spend_section(), rx.box()),
		rx.cond(DashboardState.active_mode == "risk", _risk_section(), rx.box()),
		rx.cond(DashboardState.active_mode == "forecast", _forecast_section(), rx.box()),
		spacing="4",
		width="100%",
	)


def _overview_section() -> rx.Component:
	return rx.vstack(
		_spend_section(),
		_risk_section(),
		spacing="6",
		width="100%",
	)


def _spend_section() -> rx.Component:
	return rx.vstack(
		rx.text(
			"Spend & Variance",
			font_size="14px",
			font_weight="600",
			color=FROST_WHITE,
			font_family="Inter, SF Pro, Poppins, sans-serif",
		),
		rx.box(
			chart_card(
				"sector_treemap",
				"Sector-Wise Spend Composition",
				DashboardState.sector_treemap_figure,
			),
			chart_card(
				"root_cause_variance",
				"Root Cause Variance Breakdown",
				DashboardState.root_cause_variance_figure,
				on_hover=DashboardState.explain_variance_from_hover,
			),
			chart_card(
				"trend_and_seasonality",
				"Trend & Seasonality",
				DashboardState.trend_and_seasonality_figure,
			),
			display="grid",
			grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))",
			gap="12px",
			width="100%",
		),
		predictive_insight_card(),
		spacing="3",
		width="100%",
	)


def _risk_section() -> rx.Component:
	return rx.vstack(
		rx.text(
			"Risk & Governance",
			font_size="14px",
			font_weight="600",
			color=FROST_WHITE,
			font_family="Inter, SF Pro, Poppins, sans-serif",
		),
		rx.box(
			chart_card(
				"risk_heatmap",
				"Risk Heatmap",
				DashboardState.risk_heatmap_figure,
			),
			chart_card(
				"aging_risk_histogram",
				"Aging Risk Histogram",
				DashboardState.aging_risk_histogram_figure,
			),
			display="grid",
			grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))",
			gap="12px",
			width="100%",
		),
		risk_action_list(),
		spacing="3",
		width="100%",
	)


def _forecast_section() -> rx.Component:
	return rx.box(
		rx.vstack(
			rx.heading(
				"Predictive Insight Markers",
				size="6",
				color=FROST_WHITE,
				font_family="Inter, SF Pro, Poppins, sans-serif",
			),
			rx.text(
				"Forecast mode highlights trend markers in the spend chart. Hover bars in Spend mode for Claude root-cause insight depth.",
				color=SLATE_GREY,
				font_size="13px",
				font_family="Inter, SF Pro, Poppins, sans-serif",
			),
			chart_card(
				"trend_and_seasonality",
				"Trend and Seasonality",
				DashboardState.trend_and_seasonality_figure,
			),
			spacing="3",
			align="stretch",
		),
		width="100%",
		padding="1rem",
		border_radius="12px",
		background=SOFT_GRAPHITE,
		border="1px solid #253047",
		box_shadow="0 10px 24px rgba(76, 141, 255, 0.14)",
	)
