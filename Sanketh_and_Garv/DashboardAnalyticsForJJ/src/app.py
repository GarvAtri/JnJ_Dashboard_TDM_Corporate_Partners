"""Application entrypoint for DashboardAnalyticsForJJ."""

from __future__ import annotations

import reflex as rx

from src.state import DashboardState
from src.ui.layout import dashboard_page


app = rx.App()
app.add_page(
	dashboard_page,
	route="/",
	title="DashboardAnalyticsForJJ",
	on_load=DashboardState.load_dashboard,
)
