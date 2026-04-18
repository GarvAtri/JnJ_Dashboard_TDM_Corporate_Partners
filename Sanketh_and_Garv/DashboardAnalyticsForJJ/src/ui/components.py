"""Reusable Reflex UI components — enterprise analytics dashboard.

Design system following Material/Carbon/Fluent BI UX patterns:
- 4/8/12/16/24 px spacing scale
- Consistent typography hierarchy: label → title → value
- 12-column responsive grid via CSS grid
- Compact, high-density information layout
"""

from __future__ import annotations

import reflex as rx

from src.state import DashboardState

# ═══════════════════════════════════════════════════════════════
# Design Tokens
# ═══════════════════════════════════════════════════════════════

# Palette
BG = "#0B1221"
BG_CARD = "#1E2635"
BG_INPUT = "#111A2A"
T1 = "#F5F7FA"          # primary text
T2 = "#D7E4F7"          # secondary text
T3 = "#8A93A6"          # muted / labels
T4 = "#6B7D99"          # hint
CYAN = "#3EE7E0"
BLUE = "#4C8DFF"
PURPLE = "#A66BFF"
GREEN = "#4CD964"
RED = "#FF5A5F"
AMBER = "#FFC043"
BORDER = "#253047"
BORDER_SUBTLE = "#1E2A3E"

# Typography
FONT = "Inter, SF Pro, Poppins, sans-serif"

# Spacing scale (px): 4 8 12 16 24 32
SP4 = "4px"
SP8 = "8px"
SP12 = "12px"
SP16 = "16px"
SP24 = "24px"

# Radii
R_SM = "6px"
R_MD = "10px"
R_LG = "14px"
R_PILL = "999px"

# Transitions
EASE = "all 180ms cubic-bezier(.4,0,.2,1)"


# ═══════════════════════════════════════════════════════════════
# 1. Filter Bar — unified strip with clear hierarchy
# ═══════════════════════════════════════════════════════════════


def dashboard_controls_panel() -> rx.Component:
    """Unified filter bar: date pills | dropdowns | toggle/export actions."""
    return rx.vstack(
        # ── primary row ──
        rx.hstack(
            _time_range_pills(),
            _filter_select("Sector", DashboardState.sector_options,
                           DashboardState.selected_sector, DashboardState.set_sector),
            _filter_select("PO Status", DashboardState.po_status_options,
                           DashboardState.selected_po_status, DashboardState.set_po_status),
            rx.spacer(),
            # advanced toggle — subtle icon+label button
            rx.el.button(
                rx.hstack(
                    rx.icon("sliders-horizontal", size=13),
                    rx.cond(DashboardState.show_advanced_filters, "Less", "More"),
                    spacing="1", align="center",
                ),
                on_click=DashboardState.toggle_advanced_filters,
                title="Advanced filters",
                background="transparent",
                color=T3,
                border="none",
                font_size="12px",
                font_family=FONT,
                cursor="pointer",
                padding="4px 8px",
                border_radius=R_SM,
                transition=EASE,
                _hover={"color": CYAN, "background": "rgba(62,231,224,0.06)"},
            ),
            # hide/show toggle — icon+label
            rx.el.button(
                rx.hstack(
                    rx.icon(
                        rx.cond(DashboardState.show_filters, "eye-off", "eye"),
                        size=13,
                    ),
                    rx.cond(DashboardState.show_filters, "Hide", "Filters"),
                    spacing="1", align="center",
                ),
                on_click=DashboardState.toggle_filters,
                title="Toggle filter visibility",
                background="transparent",
                color=T3,
                border="none",
                font_size="12px",
                font_family=FONT,
                cursor="pointer",
                padding="4px 8px",
                border_radius=R_SM,
                transition=EASE,
                _hover={"color": CYAN, "background": "rgba(62,231,224,0.06)"},
            ),
            # export — ghost button with icon
            rx.el.button(
                rx.hstack(
                    rx.icon("download", size=13),
                    "Export",
                    spacing="1", align="center",
                ),
                on_click=DashboardState.export_report,
                title="Export PDF summary",
                background="transparent",
                color=T3,
                border=f"1px solid {BORDER}",
                font_size="12px",
                font_weight="500",
                font_family=FONT,
                cursor="pointer",
                padding="4px 10px",
                border_radius=R_SM,
                transition=EASE,
                _hover={"color": BLUE, "border_color": BLUE,
                        "background": "rgba(76,141,255,0.06)"},
            ),
            spacing="3",
            align="center",
            width="100%",
            flex_wrap="wrap",
        ),
        # ── advanced row (animated slide) ──
        rx.box(
            rx.hstack(
                _filter_select("Addressable", DashboardState.addressable_options,
                               DashboardState.selected_addressable, DashboardState.set_addressable),
                _filter_select("Risk Status", DashboardState.risk_status_options,
                               DashboardState.selected_risk_status, DashboardState.set_risk_status),
                spacing="3",
                width="100%",
            ),
            max_height=rx.cond(
                DashboardState.show_filters & DashboardState.show_advanced_filters,
                "60px", "0px",
            ),
            opacity=rx.cond(
                DashboardState.show_filters & DashboardState.show_advanced_filters,
                "1", "0",
            ),
            overflow="hidden",
            transition="all 250ms ease-in-out",
            width="100%",
        ),
        # ── active chip summary ──
        _filter_chip_row(),
        # ── status toast ──
        rx.cond(
            DashboardState.last_action_message != "",
            rx.hstack(
                rx.icon("circle-check", size=14, color=GREEN),
                rx.text(DashboardState.last_action_message, color=GREEN, size="2"),
                spacing="2", align="center",
                background="rgba(76,217,100,0.08)",
                padding="4px 10px",
                border_radius=R_MD,
                border=f"1px solid rgba(76,217,100,0.25)",
                width="100%",
            ),
            rx.box(),
        ),
        # outer container
        width="100%",
        spacing="2",
        padding="8px 12px",
        border_radius=R_LG,
        background=BG_CARD,
        border=f"1px solid {BORDER}",
    )


def _time_range_pills() -> rx.Component:
    """Single unified date range selector — no duplicates."""
    return rx.hstack(
        _tr_pill("Today", "today"),
        _tr_pill("Week", "week"),
        _tr_pill("Month", "month"),
        spacing="1",
        padding="3px",
        background=BG_INPUT,
        border=f"1px solid {BORDER}",
        border_radius=R_PILL,
        flex_shrink="0",
    )


def _tr_pill(label: str, value: str) -> rx.Component:
    active = DashboardState.selected_time_range == value
    return rx.el.button(
        label,
        on_click=DashboardState.set_time_range(value),
        background=rx.cond(active, "rgba(62,231,224,0.18)", "transparent"),
        color=rx.cond(active, CYAN, T3),
        border=rx.cond(active, f"1px solid {CYAN}", "1px solid transparent"),
        border_radius=R_PILL,
        padding="3px 10px",
        font_size="11px",
        font_weight="600",
        font_family=FONT,
        cursor="pointer",
        white_space="nowrap",
        transition=EASE,
        _hover={"color": CYAN},
    )


def _filter_select(label: str, options, value, on_change) -> rx.Component:
    """Compact labelled dropdown — right-sized for its container."""
    return rx.hstack(
        rx.text(label, font_size="11px", color=T3, font_family=FONT,
                white_space="nowrap", flex_shrink="0"),
        rx.select(
            options,
            value=value,
            on_change=on_change,
            size="1",
            radius="medium",
        ),
        spacing="2",
        align="center",
        flex_shrink="0",
    )


def _filter_chip_row() -> rx.Component:
    return rx.cond(
        DashboardState.show_filters,
        rx.hstack(
            rx.foreach(
                DashboardState.active_filter_chips,
                lambda chip: rx.el.span(
                    chip,
                    style={
                        "background": "rgba(62,231,224,0.10)",
                        "color": CYAN,
                        "border": f"1px solid rgba(62,231,224,0.30)",
                        "border-radius": R_PILL,
                        "padding": "2px 8px",
                        "font-size": "11px",
                        "font-family": FONT,
                        "white-space": "nowrap",
                    },
                ),
            ),
            spacing="2",
            flex_wrap="wrap",
            width="100%",
        ),
        rx.box(),
    )


# keep legacy name for imports that use it
def time_range_selector(compact: bool = False) -> rx.Component:
    """Backward-compat wrapper."""
    return _time_range_pills()


# ═══════════════════════════════════════════════════════════════
# 2. KPI Band — compact horizontal row, all KPIs visible at once
# ═══════════════════════════════════════════════════════════════


def metric_card(
    title: str, value: str, accent_color: str,
    icon_symbol: str, trend_text: str,
) -> rx.Component:
    """Compact KPI tile: accent bar | label + value + trend badge."""
    is_positive = trend_text.startswith("+")
    trend_color = rx.cond(is_positive, GREEN, RED)
    trend_bg = rx.cond(is_positive,
                       "rgba(76,217,100,0.12)", "rgba(255,90,95,0.12)")
    return rx.box(
        rx.hstack(
            # accent bar
            rx.box(
                width="3px",
                height="100%",
                border_radius="2px",
                background=accent_color,
                flex_shrink="0",
            ),
            rx.vstack(
                # title + trend
                rx.hstack(
                    rx.text(
                        title, font_size="11px", color=T3,
                        text_transform="uppercase", letter_spacing="0.06em",
                        font_family=FONT, white_space="nowrap",
                    ),
                    rx.el.span(
                        trend_text,
                        style={
                            "font-size": "10px",
                            "font-weight": "600",
                            "font-family": FONT,
                            "padding": "1px 6px",
                            "border-radius": R_PILL,
                            "white-space": "nowrap",
                        },
                        color=trend_color,
                        background=trend_bg,
                    ),
                    spacing="2",
                    align="center",
                ),
                # value
                rx.text(
                    value,
                    font_size="26px",
                    line_height="1.1",
                    font_weight="700",
                    color=T1,
                    font_family=FONT,
                ),
                spacing="1",
                align="start",
                justify="center",
                flex="1",
            ),
            spacing="3",
            align="stretch",
            height="100%",
        ),
        padding="8px 12px",
        border_radius=R_MD,
        background=BG_CARD,
        border=f"1px solid {BORDER}",
        transition=EASE,
        _hover={
            "border_color": accent_color,
            "box_shadow": f"0 4px 16px rgba(62,231,224,0.10)",
        },
        height="72px",
        min_width="0",
        flex="1",
    )


# ═══════════════════════════════════════════════════════════════
# 3. Chart Card — reusable template: header | body | no Plotly title
# ═══════════════════════════════════════════════════════════════


def chart_card(
    chart_id: str,
    title: str,
    figure,
    *,
    on_hover=None,
) -> rx.Component:
    """Chart card: header (title left, expand right) | chart body.

    Plotly titles are intentionally removed — the card header is the
    single source of truth for the chart name, preventing overlap with
    legends or annotations.
    """
    chart_el = rx.cond(
        on_hover is None,
        rx.plotly(data=figure, width="100%", height="100%"),
        rx.plotly(data=figure, on_hover=on_hover, width="100%", height="100%"),
    )
    return rx.box(
        rx.vstack(
            # ── header ──
            rx.hstack(
                rx.text(
                    title, font_size="13px", font_weight="600",
                    color=T1, font_family=FONT, white_space="nowrap",
                    overflow="hidden", text_overflow="ellipsis",
                ),
                rx.spacer(),
                rx.el.button(
                    rx.icon("maximize-2", size=14),
                    on_click=DashboardState.open_chart_modal(chart_id),
                    title="Expand chart",
                    background="transparent",
                    color=T4,
                    border="none",
                    border_radius=R_SM,
                    padding="4px",
                    cursor="pointer",
                    display="inline-flex",
                    align_items="center",
                    justify_content="center",
                    transition=EASE,
                    _hover={"color": CYAN, "background": "rgba(62,231,224,0.08)"},
                ),
                align="center",
                width="100%",
                padding_bottom=SP8,
                border_bottom=f"1px solid {BORDER_SUBTLE}",
            ),
            # ── chart body ──
            rx.box(chart_el, width="100%", flex="1", min_height="0"),
            spacing="2",
            align="stretch",
            height="100%",
        ),
        background=BG_CARD,
        border=f"1px solid {BORDER}",
        border_radius=R_LG,
        padding="12px",
        transition=EASE,
        _hover={"border_color": "rgba(62,231,224,0.20)"},
        height="380px",
        min_height="380px",
        overflow="hidden",
    )


# ═══════════════════════════════════════════════════════════════
# 4. Risk Action List — compact table rows, scrollable
# ═══════════════════════════════════════════════════════════════

_RISK_SCROLL = {
    "&::-webkit-scrollbar": {"width": "4px"},
    "&::-webkit-scrollbar-track": {"background": "transparent"},
    "&::-webkit-scrollbar-thumb": {"background": "rgba(62,231,224,0.20)", "border_radius": "8px"},
    "scrollbar-width": "thin",
    "scrollbar-color": "rgba(62,231,224,0.20) transparent",
}


def risk_action_list() -> rx.Component:
    """Scrollable risk action table with fixed header and column labels."""
    return rx.vstack(
        # header
        rx.hstack(
            rx.hstack(
                rx.icon("shield-alert", size=15, color=CYAN),
                rx.text(
                    "Risk Owner Actions", font_size="13px",
                    font_weight="600", color=T1, font_family=FONT,
                ),
                spacing="2", align="center",
            ),
            rx.spacer(),
            rx.el.span(
                DashboardState.open_risks.length().to(str),
                style={
                    "font-size": "11px", "font-weight": "600",
                    "color": CYAN, "background": "rgba(62,231,224,0.10)",
                    "padding": "2px 8px", "border-radius": R_PILL,
                    "font-family": FONT,
                },
            ),
            align="center",
            width="100%",
            padding_bottom=SP8,
        ),
        # column labels
        rx.hstack(
            rx.text("Risk", font_size="10px", color=T4, width="28%",
                    text_transform="uppercase", letter_spacing="0.06em", font_family=FONT),
            rx.text("Owner", font_size="10px", color=T4, width="20%",
                    text_transform="uppercase", letter_spacing="0.06em", font_family=FONT),
            rx.text("Status", font_size="10px", color=T4, width="18%",
                    text_transform="uppercase", letter_spacing="0.06em", font_family=FONT),
            rx.text("Age", font_size="10px", color=T4, width="14%",
                    text_transform="uppercase", letter_spacing="0.06em", font_family=FONT),
            rx.text("Action", font_size="10px", color=T4, width="20%",
                    text_transform="uppercase", letter_spacing="0.06em", font_family=FONT,
                    text_align="right"),
            width="100%",
            padding="0 4px",
        ),
        rx.box(width="100%", height="1px", background=BORDER_SUBTLE),
        # scrollable rows
        rx.box(
            rx.foreach(DashboardState.open_risks, _risk_row),
            width="100%",
            display="flex",
            flex_direction="column",
            gap=SP4,
            overflow_y="auto",
            max_height="320px",
            padding_right="2px",
            **_RISK_SCROLL,
        ),
        # container
        width="100%",
        spacing="1",
        padding="12px",
        border_radius=R_LG,
        background=BG_CARD,
        border=f"1px solid {BORDER}",
    )


def _risk_row(risk: dict) -> rx.Component:
    """Single compact risk row: id+desc | owner | status badge | age | action btn."""
    return rx.hstack(
        # risk id + truncated description
        rx.vstack(
            rx.text(
                risk["risk_id"], font_size="12px", font_weight="600",
                color=T1, font_family=FONT, white_space="nowrap",
            ),
            rx.text(
                risk["risk_description"], font_size="11px", color=T3,
                font_family=FONT, white_space="nowrap",
                overflow="hidden", text_overflow="ellipsis",
                max_width="100%",
            ),
            spacing="0",
            width="28%",
            min_width="0",
        ),
        # owner
        rx.text(
            risk["risk_owner"], font_size="12px", color=T2,
            font_family=FONT, width="20%", white_space="nowrap",
            overflow="hidden", text_overflow="ellipsis",
        ),
        # status badge
        rx.el.span(
            risk["risk_status"],
            style={
                "font-size": "10px",
                "font-weight": "600",
                "padding": "2px 8px",
                "border-radius": R_PILL,
                "white-space": "nowrap",
                "font-family": FONT,
            },
            color=_status_color(risk["risk_status"]),
            background=_status_bg(risk["risk_status"]),
            width="18%",
        ),
        # days open
        rx.hstack(
            rx.text(
                risk["days_open"], font_size="12px",
                color=_age_color(risk["days_open"]),
                font_weight="600", font_family=FONT,
            ),
            rx.text("d", font_size="10px", color=T4, font_family=FONT),
            spacing="0",
            align="baseline",
            width="14%",
        ),
        # action button
        rx.box(
            rx.el.button(
                "Review",
                on_click=DashboardState.email_risk_owner(risk["risk_id"]),
                background="transparent",
                color=CYAN,
                border=f"1px solid rgba(62,231,224,0.25)",
                border_radius=R_SM,
                padding="3px 10px",
                font_size="11px",
                font_weight="600",
                font_family=FONT,
                cursor="pointer",
                transition=EASE,
                _hover={
                    "background": "rgba(62,231,224,0.08)",
                    "border_color": CYAN,
                },
            ),
            width="20%",
            text_align="right",
        ),
        align="center",
        width="100%",
        padding="6px 4px",
        border_radius=R_SM,
        transition=EASE,
        _hover={"background": "rgba(62,231,224,0.03)"},
    )


def _status_color(status) -> rx.Var:
    return rx.cond(
        status == "Critical", RED,
        rx.cond(status == "High", AMBER,
                rx.cond(status == "Open", BLUE, T3)))


def _status_bg(status) -> rx.Var:
    return rx.cond(
        status == "Critical", "rgba(255,90,95,0.12)",
        rx.cond(status == "High", "rgba(255,192,67,0.12)",
                rx.cond(status == "Open", "rgba(76,141,255,0.12)",
                        "rgba(138,147,166,0.08)")))


def _age_color(days) -> rx.Var:
    return rx.cond(
        days.to(int) > 60, RED,
        rx.cond(days.to(int) > 30, AMBER, T2))


# ═══════════════════════════════════════════════════════════════
# 5. Predictive Insight Card
# ═══════════════════════════════════════════════════════════════


def predictive_insight_card() -> rx.Component:
    """AI insight card — compact inline."""
    return rx.hstack(
        rx.el.span(
            "AI",
            style={
                "font-size": "10px", "font-weight": "700",
                "color": CYAN, "background": "rgba(62,231,224,0.14)",
                "border": f"1px solid rgba(62,231,224,0.35)",
                "border-radius": R_SM, "padding": "2px 6px",
                "font-family": FONT,
            },
        ),
        rx.vstack(
            rx.text("Claude Variance Insight", font_size="12px",
                    font_weight="600", color=T1, font_family=FONT),
            rx.text(
                DashboardState.variance_explanation,
                white_space="pre-wrap", color=T2, size="2",
                font_family=FONT, line_height="1.5",
            ),
            spacing="1", align="stretch", flex="1", min_width="0",
        ),
        spacing="3",
        align="start",
        padding="8px 12px",
        border_radius=R_MD,
        background=BG_INPUT,
        border=f"1px solid rgba(62,231,224,0.18)",
        width="100%",
    )


# ═══════════════════════════════════════════════════════════════
# 6. Mode Tabs
# ═══════════════════════════════════════════════════════════════


def analytics_mode_tabs() -> rx.Component:
    return rx.hstack(
        _mode_tab("Overview", "overview"),
        _mode_tab("Spend", "spend"),
        _mode_tab("Risk", "risk"),
        _mode_tab("Forecast", "forecast"),
        spacing="1",
        width="100%",
    )


def _mode_tab(label: str, mode: str) -> rx.Component:
    active = DashboardState.active_mode == mode
    return rx.el.button(
        label,
        on_click=DashboardState.set_active_mode(mode),
        background=rx.cond(active, "rgba(62,231,224,0.14)", "transparent"),
        color=rx.cond(active, CYAN, T3),
        border=rx.cond(active, f"1px solid {CYAN}", "1px solid transparent"),
        border_radius=R_PILL,
        padding="4px 12px",
        font_size="12px",
        font_weight="600",
        font_family=FONT,
        cursor="pointer",
        transition=EASE,
        _hover={"color": CYAN, "border_color": "rgba(62,231,224,0.30)"},
    )


# ═══════════════════════════════════════════════════════════════
# 7. Expanded Chart Modal
# ═══════════════════════════════════════════════════════════════


def expanded_chart_modal() -> rx.Component:
    """Full-screen chart overlay for detailed inspection."""
    return rx.cond(
        DashboardState.expanded_chart_id != "",
        rx.box(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            _expanded_chart_title(),
                            font_size="16px", font_weight="700",
                            color=T1, font_family=FONT,
                        ),
                        rx.spacer(),
                        rx.el.button(
                            rx.icon("x", size=16),
                            on_click=DashboardState.close_chart_modal,
                            title="Close",
                            background="transparent",
                            color=T3,
                            border="none",
                            cursor="pointer",
                            border_radius=R_SM,
                            padding="4px",
                            transition=EASE,
                            _hover={"color": RED},
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.text(
                        _expanded_chart_description(),
                        color=T3, font_size="12px", font_family=FONT,
                    ),
                    rx.box(
                        _expanded_chart_component(),
                        height="64vh",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.el.span(
                            "AI",
                            style={
                                "font-size": "10px", "font-weight": "700",
                                "color": CYAN, "background": "rgba(62,231,224,0.14)",
                                "border": f"1px solid rgba(62,231,224,0.35)",
                                "border-radius": R_SM, "padding": "2px 6px",
                                "font-family": FONT,
                            },
                        ),
                        rx.text(
                            _expanded_chart_insight(),
                            color=T2, font_size="12px", font_family=FONT,
                        ),
                        spacing="2",
                        align="start",
                        background=BG_INPUT,
                        border=f"1px solid {BORDER}",
                        border_radius=R_MD,
                        padding="8px 12px",
                        width="100%",
                    ),
                    spacing="3",
                    align="stretch",
                ),
                background=BG,
                border=f"1px solid {BORDER}",
                border_radius=R_LG,
                padding="16px",
                width="min(1200px, 92vw)",
                height="90vh",
                box_shadow="0 24px 64px rgba(0,0,0,0.55)",
            ),
            position="fixed",
            inset="0",
            display="flex",
            align_items="center",
            justify_content="center",
            background="rgba(6,10,18,0.78)",
            z_index="60",
        ),
        rx.box(),
    )


# ── modal helpers ──

def _expanded_chart_title() -> rx.Var:
    return rx.cond(
        DashboardState.expanded_chart_id == "sector_treemap",
        "Sector-Wise Spend Composition",
        rx.cond(
            DashboardState.expanded_chart_id == "root_cause_variance",
            "Root Cause Variance Breakdown",
            rx.cond(
                DashboardState.expanded_chart_id == "trend_and_seasonality",
                "Trend and Seasonality",
                rx.cond(
                    DashboardState.expanded_chart_id == "risk_heatmap",
                    "Risk Heatmap",
                    "Aging Risk Histogram",
                ),
            ),
        ),
    )


def _expanded_chart_description() -> rx.Var:
    return rx.cond(
        DashboardState.expanded_chart_id == "sector_treemap",
        "Sector-level spend concentration and share proportions.",
        rx.cond(
            DashboardState.expanded_chart_id == "root_cause_variance",
            "Grouped variance drivers with axis context for root-cause interpretation.",
            rx.cond(
                DashboardState.expanded_chart_id == "trend_and_seasonality",
                "Monthly spend trend, cumulative variance, and predictive markers.",
                rx.cond(
                    DashboardState.expanded_chart_id == "risk_heatmap",
                    "5\u00d75 severity matrix with non-overlapping labels.",
                    "Unresolved risk age distribution with full bins.",
                ),
            ),
        ),
    )


def _expanded_chart_insight() -> rx.Var:
    return rx.cond(
        DashboardState.expanded_chart_id == "root_cause_variance",
        DashboardState.variance_explanation,
        rx.cond(
            DashboardState.expanded_chart_id == "trend_and_seasonality",
            "Review the predictive marker trajectory against monthly volatility.",
            "Use full-scale labels to validate concentration and outliers.",
        ),
    )


def _expanded_chart_component() -> rx.Component:
    return rx.cond(
        DashboardState.expanded_chart_id == "sector_treemap",
        rx.plotly(data=DashboardState.sector_treemap_figure, width="100%", height="100%"),
        rx.cond(
            DashboardState.expanded_chart_id == "root_cause_variance",
            rx.plotly(data=DashboardState.root_cause_variance_figure, width="100%", height="100%"),
            rx.cond(
                DashboardState.expanded_chart_id == "trend_and_seasonality",
                rx.plotly(data=DashboardState.trend_and_seasonality_figure, width="100%", height="100%"),
                rx.cond(
                    DashboardState.expanded_chart_id == "risk_heatmap",
                    rx.plotly(data=DashboardState.risk_heatmap_figure, width="100%", height="100%"),
                    rx.plotly(data=DashboardState.aging_risk_histogram_figure, width="100%", height="100%"),
                ),
            ),
        ),
    )


# ═══════════════════════════════════════════════════════════════
# Legacy aliases (backward compat for any other imports)
# ═══════════════════════════════════════════════════════════════

# risk_owner_card — keep for tests or other callers
def risk_owner_card(risk: dict) -> rx.Component:
    """Legacy single risk card — wraps the compact row."""
    return _risk_row(risk)


# ═══════════════════════════════════════════════════════════════
# File Upload Component
# ═══════════════════════════════════════════════════════════════

def _file_type_badge(file_type: rx.Var[str]) -> rx.Component:
    """Colored badge indicating file type."""
    return rx.box(
        rx.text(file_type, font_size="10px", font_weight="700", color=BG),
        padding="2px 8px",
        border_radius=R_PILL,
        background=rx.cond(
            file_type == "PDF",
            RED,
            rx.cond(file_type == "JSON", AMBER, GREEN),
        ),
    )


def _uploaded_file_row(file: rx.Var[dict]) -> rx.Component:
    """Single row in the uploaded files list."""
    return rx.hstack(
        _file_type_badge(file["type"]),
        rx.vstack(
            rx.text(file["name"], color=T1, font_size="13px", font_weight="500", font_family=FONT),
            rx.text(file["size"], color=T3, font_size="11px", font_family=FONT),
            spacing="0",
        ),
        rx.spacer(),
        rx.text(file["timestamp"], color=T4, font_size="11px", font_family=FONT),
        rx.el.button(
            rx.icon("x", size=14),
            on_click=DashboardState.remove_uploaded_file(file["name"]),
            background="transparent",
            color=T3,
            border="none",
            cursor="pointer",
            padding="4px",
            border_radius=R_SM,
            _hover={"color": RED, "background": "rgba(255,90,95,0.1)"},
        ),
        align="center",
        spacing="3",
        width="100%",
        padding="8px 12px",
        border_radius=R_MD,
        background=BG_INPUT,
        border=f"1px solid {BORDER_SUBTLE}",
        transition=EASE,
        _hover={"border_color": BORDER},
    )


def _column_tag(col_name: rx.Var[str]) -> rx.Component:
    """Small tag for a column name."""
    return rx.box(
        rx.text(col_name, font_size="11px", color=CYAN, font_family=FONT),
        padding="2px 8px",
        border_radius=R_PILL,
        background="rgba(62,231,224,0.08)",
        border=f"1px solid rgba(62,231,224,0.25)",
        display="inline-block",
    )


def _file_insight_card(insight: rx.Var[dict]) -> rx.Component:
    """Render a card showing scraped metadata for one file."""
    return rx.vstack(
        # Header: type badge + name + summary
        rx.hstack(
            _file_type_badge(insight["type"]),
            rx.text(insight["name"], color=T1, font_size="13px",
                    font_weight="600", font_family=FONT),
            spacing="2", align="center",
        ),
        rx.text(insight["summary"], color=T2, font_size="12px", font_family=FONT),
        # Column names
        rx.cond(
            insight["column_names"].length() > 0,  # type: ignore[attr-defined]
            rx.vstack(
                rx.text("Columns / Fields", color=T3, font_size="11px",
                        font_weight="600", font_family=FONT, letter_spacing="0.04em"),
                rx.box(
                    rx.foreach(insight["column_names"], _column_tag),
                    display="flex",
                    flex_wrap="wrap",
                    gap="6px",
                ),
                spacing="2", width="100%",
            ),
            rx.box(),
        ),
        # Headings (PDF)
        rx.cond(
            insight["type"] == "PDF",
            rx.cond(
                insight["headings"].length() > 0,  # type: ignore[attr-defined]
                rx.vstack(
                    rx.text("Headings Detected", color=T3, font_size="11px",
                            font_weight="600", font_family=FONT, letter_spacing="0.04em"),
                    rx.vstack(
                        rx.foreach(
                            insight["headings"],
                            lambda h: rx.text(h, color=T2, font_size="12px", font_family=FONT),
                        ),
                        spacing="1", width="100%",
                        max_height="120px", overflow_y="auto",
                    ),
                    spacing="2", width="100%",
                ),
                rx.box(),
            ),
            rx.box(),
        ),
        spacing="3",
        width="100%",
        padding="12px 16px",
        border_radius=R_MD,
        background=BG_INPUT,
        border=f"1px solid {BORDER}",
    )


# ═══════════════════════════════════════════════════════════════
# Claude Column Relevance Report
# ═══════════════════════════════════════════════════════════════

_CATEGORY_COLORS = {
    "spend": GREEN,
    "variance": PURPLE,
    "risk": RED,
    "time": BLUE,
    "identifier": CYAN,
    "metadata": AMBER,
    "irrelevant": T4,
}

_CATEGORY_ICONS = {
    "spend": "dollar-sign",
    "variance": "trending-up",
    "risk": "shield-alert",
    "time": "clock",
    "identifier": "key",
    "metadata": "tag",
    "irrelevant": "x-circle",
}


def _category_badge(category: rx.Var[str]) -> rx.Component:
    """Colored pill for a column category."""
    return rx.box(
        rx.text(category, font_size="10px", font_weight="700",
                text_transform="uppercase", letter_spacing="0.04em"),
        padding="2px 10px",
        border_radius=R_PILL,
        color=BG,
        background=rx.match(
            category,
            ("spend", GREEN),
            ("variance", PURPLE),
            ("risk", RED),
            ("time", BLUE),
            ("identifier", CYAN),
            ("metadata", AMBER),
            ("irrelevant", T4),
            T3,
        ),
    )


def _confidence_dot(confidence: rx.Var[str]) -> rx.Component:
    """Small dot indicating confidence level."""
    return rx.box(
        width="8px", height="8px",
        border_radius="50%",
        background=rx.match(
            confidence,
            ("high", GREEN),
            ("medium", AMBER),
            ("low", RED),
            T4,
        ),
        title=confidence,
    )


def _column_relevance_row(col: rx.Var[dict]) -> rx.Component:
    """Single row in the relevance report."""
    return rx.hstack(
        _category_badge(col["category"]),
        _confidence_dot(col["confidence"]),
        rx.vstack(
            rx.hstack(
                rx.text(col["name"], color=T1, font_size="13px",
                        font_weight="500", font_family=FONT),
                rx.cond(
                    col["maps_to"],
                    rx.text(
                        rx.text.span("→ ", color=T4),
                        rx.text.span(col["maps_to"], color=CYAN),
                        font_size="11px", font_family=FONT,
                    ),
                    rx.box(),
                ),
                spacing="2", align="center",
            ),
            rx.text(col["reason"], color=T3, font_size="11px", font_family=FONT),
            spacing="0",
        ),
        align="center",
        spacing="3",
        width="100%",
        padding="6px 10px",
        border_radius=R_SM,
        transition=EASE,
        _hover={"background": "rgba(62,231,224,0.04)"},
    )


def _column_relevance_report_panel() -> rx.Component:
    """Full relevance report panel showing Claude's analysis."""
    report = DashboardState.column_relevance_report
    return rx.vstack(
        # Header
        rx.hstack(
            rx.hstack(
                rx.icon("brain", size=16, color=PURPLE),
                rx.text("Claude AI Relevance Report", color=T1, font_size="14px",
                        font_weight="600", font_family=FONT),
                spacing="2", align="center",
            ),
            rx.spacer(),
            rx.box(
                rx.text(
                    report["source"],  # type: ignore[index]
                    font_size="10px", color=PURPLE, font_family=FONT,
                    text_transform="uppercase", letter_spacing="0.04em",
                ),
                padding="2px 8px",
                border_radius=R_PILL,
                background="rgba(166,107,255,0.12)",
                border="1px solid rgba(166,107,255,0.3)",
            ),
            rx.el.button(
                rx.icon("x", size=14),
                on_click=DashboardState.clear_column_analysis,
                background="transparent",
                color=T3, border="none", cursor="pointer",
                _hover={"color": T1},
            ),
            width="100%", align="center",
        ),
        # Summary
        rx.text(
            report["file_summary"],  # type: ignore[index]
            color=T2, font_size="12px", font_family=FONT
        ),
        # Stats strip
        rx.hstack(
            rx.box(
                rx.vstack(
                    rx.text(report["relevant_count"],  # type: ignore[index]
                            color=GREEN, font_size="20px", font_weight="700", font_family=FONT),
                    rx.text("Relevant", color=T3, font_size="10px", font_family=FONT),
                    spacing="0", align="center",
                ),
                padding="8px 16px",
                border_radius=R_MD,
                background="rgba(76,217,100,0.08)",
                border=f"1px solid rgba(76,217,100,0.2)",
                flex="1",
                text_align="center",
            ),
            rx.box(
                rx.vstack(
                    rx.text(report["irrelevant_count"],  # type: ignore[index]
                            color=T4, font_size="20px", font_weight="700", font_family=FONT),
                    rx.text("Irrelevant", color=T3, font_size="10px", font_family=FONT),
                    spacing="0", align="center",
                ),
                padding="8px 16px",
                border_radius=R_MD,
                background="rgba(107,125,153,0.08)",
                border=f"1px solid rgba(107,125,153,0.2)",
                flex="1",
                text_align="center",
            ),
            spacing="3", width="100%",
        ),
        # Column list
        rx.vstack(
            rx.foreach(report["columns"], _column_relevance_row),  # type: ignore[index]
            spacing="1",
            width="100%",
            max_height="300px",
            overflow_y="auto",
        ),
        # Recommendation
        rx.box(
            rx.hstack(
                rx.icon("lightbulb", size=14, color=AMBER),
                rx.text("Recommendation", color=AMBER, font_size="11px",
                        font_weight="600", font_family=FONT),
                spacing="2", align="center",
            ),
            rx.text(
                report["recommendation"],  # type: ignore[index]
                color=T2, font_size="12px", font_family=FONT,
            ),
            padding="10px 14px",
            border_radius=R_MD,
            background="rgba(255,192,67,0.06)",
            border=f"1px solid rgba(255,192,67,0.2)",
            width="100%",
        ),
        spacing="4",
        width="100%",
        padding="16px",
        border_radius=R_LG,
        background=BG_CARD,
        border=f"1px solid {BORDER}",
    )


def file_upload_panel() -> rx.Component:
    """Upload modal for Excel, JSON, and PDF files."""
    return rx.cond(
        DashboardState.show_upload_modal,
        rx.box(
            rx.box(
                rx.vstack(
                    # ── Header ──
                    rx.hstack(
                        rx.hstack(
                            rx.icon("upload", size=18, color=CYAN),
                            rx.text("Upload Files", color=T1, font_size="16px",
                                    font_weight="600", font_family=FONT),
                            spacing="2", align="center",
                        ),
                        rx.spacer(),
                        rx.el.button(
                            rx.icon("x", size=16),
                            on_click=DashboardState.toggle_upload_modal,
                            background="transparent",
                            color=T3, border="none", cursor="pointer",
                            _hover={"color": T1},
                        ),
                        width="100%", align="center",
                    ),
                    rx.text(
                        "Drag & drop or click to upload Excel (.xlsx/.xls), JSON, or PDF files.",
                        color=T3, font_size="12px", font_family=FONT,
                    ),
                    # ── Drop zone ──
                    rx.upload(
                        rx.vstack(
                            rx.icon("folder-up", size=36, color=CYAN, opacity="0.7"),
                            rx.text("Drop files here or click to browse",
                                    color=T2, font_size="13px", font_family=FONT),
                            rx.text(".xlsx  ·  .xls  ·  .json  ·  .pdf",
                                    color=T4, font_size="11px", font_family=FONT),
                            spacing="2", align="center", justify="center",
                            padding="2rem",
                        ),
                        id="file_upload",
                        multiple=True,
                        accept={
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
                            "application/vnd.ms-excel": [".xls"],
                            "application/json": [".json"],
                            "application/pdf": [".pdf"],
                        },
                        border=f"2px dashed {BORDER}",
                        border_radius=R_LG,
                        background=BG_INPUT,
                        cursor="pointer",
                        width="100%",
                        transition=EASE,
                        _hover={"border_color": CYAN, "background": "rgba(62,231,224,0.04)"},
                    ),
                    # ── Upload button ──
                    rx.hstack(
                        rx.el.button(
                            rx.hstack(
                                rx.icon("upload", size=14),
                                "Upload Selected",
                                spacing="2", align="center",
                            ),
                            on_click=DashboardState.handle_upload(rx.upload_files(upload_id="file_upload")),
                            background=CYAN,
                            color=BG,
                            border="none",
                            font_weight="600",
                            font_size="13px",
                            font_family=FONT,
                            padding="8px 20px",
                            border_radius=R_MD,
                            cursor="pointer",
                            transition=EASE,
                            _hover={"opacity": "0.85", "transform": "translateY(-1px)"},
                        ),
                        rx.cond(
                            DashboardState.uploaded_files.length() > 0,  # type: ignore[attr-defined]
                            rx.el.button(
                                rx.hstack(
                                    rx.icon("trash-2", size=14),
                                    "Clear All",
                                    spacing="2", align="center",
                                ),
                                on_click=DashboardState.clear_all_uploads,
                                background="transparent",
                                color=RED,
                                border=f"1px solid {RED}",
                                font_size="12px",
                                font_family=FONT,
                                padding="6px 14px",
                                border_radius=R_MD,
                                cursor="pointer",
                                transition=EASE,
                                _hover={"background": "rgba(255,90,95,0.1)"},
                            ),
                            rx.box(),
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    # ── Progress bar ──
                    rx.cond(
                        DashboardState.is_uploading,
                        rx.box(
                            rx.box(
                                width=DashboardState.upload_progress.to(str) + "%",  # type: ignore[attr-defined]
                                height="4px",
                                background=CYAN,
                                border_radius=R_PILL,
                                transition="width 300ms ease",
                            ),
                            width="100%",
                            height="4px",
                            background=BORDER,
                            border_radius=R_PILL,
                        ),
                        rx.box(),
                    ),
                    # ── Error message ──
                    rx.cond(
                        DashboardState.upload_error != "",
                        rx.box(
                            rx.text(DashboardState.upload_error, color=RED,
                                    font_size="12px", font_family=FONT),
                            background="rgba(255,90,95,0.08)",
                            border=f"1px solid {RED}",
                            border_radius=R_MD,
                            padding="8px 12px",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    # ── File list ──
                    rx.cond(
                        DashboardState.uploaded_files.length() > 0,  # type: ignore[attr-defined]
                        rx.vstack(
                            rx.hstack(
                                rx.text("Uploaded Files", color=T2, font_size="13px",
                                        font_weight="600", font_family=FONT),
                                rx.spacer(),
                                rx.text(
                                    DashboardState.uploaded_files.length().to(str) + " file(s)",  # type: ignore[attr-defined]
                                    color=T4, font_size="11px", font_family=FONT,
                                ),
                                width="100%", align="center",
                            ),
                            rx.vstack(
                                rx.foreach(DashboardState.uploaded_files, _uploaded_file_row),
                                spacing="2",
                                width="100%",
                                max_height="220px",
                                overflow_y="auto",
                            ),
                            # ── Analyze button ──
                            rx.el.button(
                                rx.hstack(
                                    rx.icon("scan-search", size=14),
                                    rx.cond(
                                        DashboardState.is_scraping,
                                        "Analyzing...",
                                        "Analyze Files",
                                    ),
                                    spacing="2", align="center",
                                ),
                                on_click=DashboardState.scrape_uploaded_files,
                                disabled=DashboardState.is_scraping,
                                background=BLUE,
                                color=T1,
                                border="none",
                                font_weight="600",
                                font_size="13px",
                                font_family=FONT,
                                padding="8px 20px",
                                border_radius=R_MD,
                                cursor="pointer",
                                width="100%",
                                transition=EASE,
                                _hover={"opacity": "0.85", "transform": "translateY(-1px)"},
                                _disabled={"opacity": "0.5", "cursor": "not-allowed"},
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    # ── Scrape error ──
                    rx.cond(
                        DashboardState.scrape_error != "",
                        rx.box(
                            rx.text(DashboardState.scrape_error, color=AMBER,
                                    font_size="12px", font_family=FONT),
                            background="rgba(255,192,67,0.08)",
                            border=f"1px solid {AMBER}",
                            border_radius=R_MD,
                            padding="8px 12px",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    # ── File insights results ──
                    rx.cond(
                        DashboardState.file_insights.length() > 0,  # type: ignore[attr-defined]
                        rx.vstack(
                            rx.hstack(
                                rx.icon("file-search", size=16, color=CYAN),
                                rx.text("File Insights", color=T1, font_size="14px",
                                        font_weight="600", font_family=FONT),
                                rx.spacer(),
                                rx.el.button(
                                    rx.icon("x", size=14),
                                    on_click=DashboardState.clear_file_insights,
                                    background="transparent",
                                    color=T3, border="none", cursor="pointer",
                                    _hover={"color": T1},
                                ),
                                width="100%", align="center",
                            ),
                            rx.vstack(
                                rx.foreach(DashboardState.file_insights, _file_insight_card),
                                spacing="3",
                                width="100%",
                                max_height="350px",
                                overflow_y="auto",
                            ),
                            # ── Analyze with Claude button ──
                            rx.el.button(
                                rx.hstack(
                                    rx.icon("brain", size=14),
                                    rx.cond(
                                        DashboardState.is_analyzing_columns,
                                        "Claude is analyzing...",
                                        "Analyze Relevance with Claude AI",
                                    ),
                                    spacing="2", align="center",
                                ),
                                on_click=DashboardState.analyze_columns_with_claude,
                                disabled=DashboardState.is_analyzing_columns,
                                background=PURPLE,
                                color=T1,
                                border="none",
                                font_weight="600",
                                font_size="13px",
                                font_family=FONT,
                                padding="8px 20px",
                                border_radius=R_MD,
                                cursor="pointer",
                                width="100%",
                                transition=EASE,
                                _hover={"opacity": "0.85", "transform": "translateY(-1px)"},
                                _disabled={"opacity": "0.5", "cursor": "not-allowed"},
                            ),
                            spacing="3",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    # ── Column analysis error ──
                    rx.cond(
                        DashboardState.column_analysis_error != "",
                        rx.box(
                            rx.text(DashboardState.column_analysis_error, color=RED,
                                    font_size="12px", font_family=FONT),
                            background="rgba(255,90,95,0.08)",
                            border=f"1px solid {RED}",
                            border_radius=R_MD,
                            padding="8px 12px",
                            width="100%",
                        ),
                        rx.box(),
                    ),
                    # ── Claude Relevance Report ──
                    rx.cond(
                        DashboardState.column_relevance_report.contains("columns"),  # type: ignore[attr-defined]
                        _column_relevance_report_panel(),
                        rx.box(),
                    ),
                    spacing="4",
                    align="stretch",
                ),
                background=BG,
                border=f"1px solid {BORDER}",
                border_radius=R_LG,
                padding="24px",
                width="min(740px, 94vw)",
                max_height="85vh",
                overflow_y="auto",
                box_shadow="0 24px 64px rgba(0,0,0,0.55)",
                on_click=rx.stop_propagation,
            ),
            position="fixed",
            inset="0",
            display="flex",
            align_items="center",
            justify_content="center",
            background="rgba(6,10,18,0.78)",
            z_index="70",
            on_click=DashboardState.toggle_upload_modal,
        ),
        rx.box(),
    )
