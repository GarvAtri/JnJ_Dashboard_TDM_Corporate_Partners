"""Plotly chart builders for DashboardAnalyticsForJJ."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


STATUS_COLORS = {
	"Open": "#3EE7E0",
	"Partially Received": "#FFC043",
	"Closed": "#8A93A6",
}
RISK_SCALE = [(0.0, "#4CD964"), (0.5, "#FFC043"), (1.0, "#FF5A5F")]
CHART_BACKGROUND = "#1E2635"
PLOT_BACKGROUND = "#111A2A"
CYAN_GRADIENT = ["#3EE7E0", "#1FB7B0"]
BLUE_GRADIENT = ["#4C8DFF", "#2A6BE0"]
PURPLE_GRADIENT = ["#A66BFF", "#7A3CE0"]


def build_sector_treemap(data: pd.DataFrame, filters: dict[str, str]) -> go.Figure:
	"""Build the sector-wise spend composition donut chart."""
	by_sector = data.groupby("Sector", as_index=False)["PO_Total_Amount"].sum()
	figure = px.pie(
		by_sector,
		names="Sector",
		values="PO_Total_Amount",
		color_discrete_sequence=CYAN_GRADIENT + BLUE_GRADIENT + PURPLE_GRADIENT,
	)
	figure.update_traces(
		hole=0.58,
		hovertemplate="Sector=%{label}<br>Spend=$%{value:,.0f}<br>Share=%{percent}<extra></extra>",
		textposition="inside",
		textfont={"size": 11, "color": "#F5F7FA"},
		marker={"line": {"color": "#0B1221", "width": 2}},
		showlegend=True,
	)
	return _apply_layout(figure, filters)


def build_root_cause_variance_bar(data: pd.DataFrame, filters: dict[str, str]) -> go.Figure:
	"""Build the root-cause variance bar chart."""
	ordered = data.copy()
	figure = px.bar(
		ordered,
		x="Variance_vs_Budget",
		y="Root_Cause",
		color="Sector",
		orientation="h",
		color_discrete_sequence=BLUE_GRADIENT + PURPLE_GRADIENT + CYAN_GRADIENT,
	)
	figure.update_layout(barmode="group")
	figure.update_traces(
		marker={"line": {"width": 0}, "cornerradius": 8},
		hovertemplate="Root Cause=%{y}<br>Sector=%{fullData.name}<br>Variance=$%{x:,.0f}<extra></extra>",
		showlegend=True,
	)
	return _apply_layout(figure, filters)


def build_trend_and_seasonality_line(data: pd.DataFrame, filters: dict[str, str]) -> go.Figure:
	"""Build the spend and cumulative variance trend chart."""
	figure = go.Figure()
	figure.add_trace(
		go.Scatter(
			x=data["Last_Updated_Timestamp"],
			y=data["Monthly_Spend"],
			mode="lines+markers",
			name="Monthly Spend",
			line={"color": "#4C8DFF", "width": 3, "shape": "spline", "smoothing": 1.1},
			marker={"size": 7, "color": "#4C8DFF", "line": {"color": "#3EE7E0", "width": 1}},
			fill="tozeroy",
			fillcolor="rgba(76, 141, 255, 0.18)",
			hovertemplate="Month=%{x|%b %Y}<br>Monthly Spend=$%{y:,.0f}<extra></extra>",
			yaxis="y1",
		)
	)
	figure.add_trace(
		go.Scatter(
			x=data["Last_Updated_Timestamp"],
			y=data["Cumulative_Variance"],
			mode="lines+markers",
			name="Cumulative Variance",
			line={"color": "#A66BFF", "width": 3, "shape": "spline", "smoothing": 1.0},
			marker={"size": 7, "color": "#A66BFF", "line": {"color": "#3EE7E0", "width": 1}},
			hovertemplate="Month=%{x|%b %Y}<br>Cumulative Variance=$%{y:,.0f}<extra></extra>",
			yaxis="y2",
		)
	)

	if len(data) >= 3:
		recent = data.tail(3)
		x_forecast = recent["Last_Updated_Timestamp"].iloc[-1] + pd.offsets.MonthBegin(1)
		slope = (recent["Monthly_Spend"].iloc[-1] - recent["Monthly_Spend"].iloc[0]) / max(len(recent) - 1, 1)
		y_forecast = recent["Monthly_Spend"].iloc[-1] + slope
		figure.add_trace(
			go.Scatter(
				x=[recent["Last_Updated_Timestamp"].iloc[-1], x_forecast],
				y=[recent["Monthly_Spend"].iloc[-1], y_forecast],
				mode="lines+markers",
				name="Predictive Marker",
				line={"color": "#3EE7E0", "width": 2, "dash": "dot"},
				marker={"size": 8, "symbol": "diamond"},
				hovertemplate="Projection=%{x|%b %Y}<br>Forecast Spend=$%{y:,.0f}<extra></extra>",
				yaxis="y1",
			)
		)

	figure.update_layout(
		yaxis={"title": "Monthly Spend", "showgrid": True, "gridcolor": "rgba(138, 147, 166, 0.2)"},
		yaxis2={
			"title": "Cumulative Variance",
			"overlaying": "y",
			"side": "right",
			"showgrid": False,
		},
		showlegend=True,
	)
	return _apply_layout(figure, filters)


def build_risk_heatmap(data: pd.DataFrame, filters: dict[str, str]) -> go.Figure:
	"""Build the likelihood-impact heatmap for risks."""
	pivot = data.pivot(index="Likelihood_Score", columns="Impact_Score", values="Risk_Count").fillna(0)
	pivot = pivot.reindex(index=[1, 2, 3, 4, 5], columns=[1, 2, 3, 4, 5], fill_value=0)
	figure = px.imshow(
		pivot,
		labels={"x": "Impact", "y": "Likelihood", "color": "Risk Count"},
		x=[1, 2, 3, 4, 5],
		y=[1, 2, 3, 4, 5],
		text_auto=True,
		aspect="auto",
		color_continuous_scale=RISK_SCALE,
	)
	figure.update_traces(hovertemplate="Likelihood=%{y}<br>Impact=%{x}<br>Risks=%{z}<extra></extra>")
	figure.update_layout(coloraxis_colorbar={"title": "Risk Count"})
	return _apply_layout(figure, filters)


def build_aging_risk_histogram(data: pd.DataFrame, filters: dict[str, str]) -> go.Figure:
	"""Build the active risk aging histogram."""
	figure = px.histogram(
		data,
		x="Days_Open",
		nbins=12,
		color_discrete_sequence=["#4C8DFF"],
	)
	figure.update_traces(
		marker={"line": {"color": "#0B1221", "width": 1}, "cornerradius": 6},
		hovertemplate="Days Open=%{x}<br>Risk Count=%{y}<extra></extra>",
		name="Risk Aging",
		showlegend=True,
	)
	return _apply_layout(figure, filters)


def _apply_layout(figure: go.Figure, filters: dict[str, str]) -> go.Figure:
	figure.update_layout(
		paper_bgcolor=CHART_BACKGROUND,
		plot_bgcolor=PLOT_BACKGROUND,
		margin={"l": 40, "r": 20, "t": 24, "b": 40},
		font={"family": "Inter, SF Pro, Poppins, sans-serif", "size": 13, "color": "#F5F7FA"},
		showlegend=True,
		legend={
			"orientation": "h",
			"yanchor": "bottom",
			"y": 1.01,
			"xanchor": "left",
			"x": 0,
			"font": {"size": 11, "color": "#D7E4F7"},
			"bgcolor": "rgba(0,0,0,0)",
			"itemwidth": 30,
		},
		title=None,
		hoverlabel={"bgcolor": "#0B1221", "bordercolor": "#3EE7E0", "font": {"color": "#F5F7FA"}},
		transition={"duration": 260, "easing": "cubic-in-out"},
	)
	figure.update_xaxes(showgrid=True, gridcolor="rgba(138, 147, 166, 0.16)", zeroline=False)
	figure.update_yaxes(showgrid=True, gridcolor="rgba(138, 147, 166, 0.16)", zeroline=False)
	return figure
