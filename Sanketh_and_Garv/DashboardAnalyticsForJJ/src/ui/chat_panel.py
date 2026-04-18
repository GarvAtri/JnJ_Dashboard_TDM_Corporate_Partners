"""Claude chatbot panel \u2013 futuristic dark-theme redesign."""

from __future__ import annotations

import reflex as rx

from src.state import DashboardState

# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Design Tokens
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550

_BG = "#0B1221"
_BG_INPUT = "#111A2A"
_BG_USER = "#163B70"
_BG_ASST = "#1A2536"
_BG_SAVED = "#0F2A3C"
_BG_INSTR = "#141E30"

_CYAN = "#3EE7E0"
_BLUE = "#4C8DFF"
_RED = "#FF5A5F"

_T1 = "#F5F7FA"
_T2 = "#D7E4F7"
_T3 = "#9BB3D1"
_T4 = "#6B7D99"

_BORDER = "#2A354A"
_BCYAN = "rgba(62,231,224,0.25)"
_BBLUE = "rgba(76,141,255,0.25)"

_FONT = "Inter, SF Pro, Poppins, sans-serif"
_EASE = "all 220ms cubic-bezier(.4,0,.2,1)"

_R_SM = "8px"
_R_MD = "12px"
_R_LG = "16px"
_R_PILL = "999px"

_GLOW_CYAN = "0 0 12px rgba(62,231,224,0.15)"

# Scrollbar
_SCROLLBAR = {
	"&::-webkit-scrollbar": {"width": "5px"},
	"&::-webkit-scrollbar-track": {"background": "transparent"},
	"&::-webkit-scrollbar-thumb": {"background": _BCYAN, "border_radius": "8px"},
	"&::-webkit-scrollbar-thumb:hover": {"background": "rgba(62,231,224,0.45)"},
	"scrollbar-width": "thin",
	"scrollbar-color": f"{_BCYAN} transparent",
}

# Markdown rendering map
_MD_MAP = {
	"p": lambda t: rx.text(t, color=_T2, size="3", line_height="1.6", margin_bottom="0.25rem"),
	"h1": lambda t: rx.heading(t, size="4", color=_CYAN, margin_bottom="0.25rem"),
	"h2": lambda t: rx.heading(t, size="3", color=_CYAN, margin_bottom="0.2rem"),
	"h3": lambda t: rx.heading(t, size="2", color=_BLUE, margin_bottom="0.15rem"),
	"strong": lambda t: rx.el.strong(t, style={"color": _T1, "font_weight": "700"}),
	"em": lambda t: rx.el.em(t, style={"color": _T3}),
	"li": lambda t: rx.el.li(t, color=_T2, font_size="13px", line_height="1.5", margin_left="0.5rem"),
	"ul": lambda c: rx.el.ul(c, padding_left="1rem", margin_bottom="0.25rem"),
	"ol": lambda c: rx.el.ol(c, padding_left="1rem", margin_bottom="0.25rem"),
	"code": lambda t: rx.code(
		t, color=_CYAN, background="rgba(62,231,224,0.08)",
		padding="0.1rem 0.3rem", border_radius="4px", font_size="12px",
	),
}


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Helpers
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550


def _tb(icon: str, handler, tip: str, active) -> rx.Component:
	"""Tiny toolbar icon-button."""
	return rx.el.button(
		rx.icon(icon, size=14),
		on_click=handler,
		title=tip,
		background=rx.cond(active, "rgba(62,231,224,0.14)", "transparent"),
		color=rx.cond(active, _CYAN, _T4),
		border="none",
		border_radius=_R_SM,
		padding="0.3rem",
		cursor="pointer",
		display="inline-flex",
		align_items="center",
		justify_content="center",
		transition=_EASE,
		_hover={"color": _CYAN, "background": "rgba(62,231,224,0.08)"},
	)


def _nudge_btn(icon: str, dx: int, dy: int) -> rx.Component:
	"""Small arrow button for floating-mode repositioning."""
	return rx.el.button(
		rx.icon(icon, size=10),
		on_click=DashboardState.nudge_chat(dx, dy),
		background="transparent",
		color=_T4,
		border="none",
		border_radius="4px",
		padding="0.2rem",
		cursor="pointer",
		display="inline-flex",
		align_items="center",
		justify_content="center",
		_hover={"color": _T3},
	)


# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
# Main Panel
# \u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550


def chat_panel() -> rx.Component:
	"""Render the redesigned Claude chat panel."""
	return rx.box(
		# auto-scroll trigger
		rx.el.span(
			rx.script(
				"setTimeout(function(){var e=document.getElementById('chat-scroll');"
				"if(e)e.scrollTop=e.scrollHeight;},80);"
			),
			key=DashboardState.chat_scroll_counter.to(str),
			display="none",
		),
		rx.el.div(
			_header_bar(),
			rx.cond(
				DashboardState.chat_panel_collapsed,
				_collapsed_strip(),
				rx.el.div(
					_float_controls(),
					_instruction_block(),
					_chat_history(),
					_error_block(),
					_input_zone(),
					display="flex",
					flex_direction="column",
					gap="0.5rem",
					flex="1 1 0",
					min_height="0",
					overflow="hidden",
				),
			),
			display="flex",
			flex_direction="column",
			height="100%",
			gap="0.5rem",
			overflow="hidden",
		),
		# outer container
		width=rx.cond(DashboardState.chat_panel_collapsed, "56px", "420px"),
		min_width=rx.cond(DashboardState.chat_panel_collapsed, "56px", "420px"),
		max_width=rx.cond(DashboardState.chat_panel_collapsed, "56px", "420px"),
		height=rx.cond(DashboardState.chat_panel_mode == "floating", "78vh", "100vh"),
		padding=rx.cond(DashboardState.chat_panel_collapsed, "0.5rem 0.25rem", "0.75rem"),
		background=_BG,
		border=rx.cond(
			DashboardState.chat_panel_mode == "floating",
			f"1px solid {_BORDER}", "none",
		),
		box_shadow=rx.cond(
			DashboardState.chat_panel_mode == "floating",
			"0 12px 48px rgba(0,0,0,0.55), 0 0 0 1px rgba(62,231,224,0.08)",
			rx.cond(
				DashboardState.chat_panel_mode == "right",
				f"inset 1px 0 0 {_BORDER}",
				f"inset -1px 0 0 {_BORDER}",
			),
		),
		border_radius=rx.cond(
			DashboardState.chat_panel_mode == "floating", _R_LG, "0",
		),
		overflow="hidden",
		display="flex",
		flex_direction="column",
		transition=_EASE,
		position=rx.cond(
			DashboardState.chat_panel_mode == "floating", "fixed", "relative",
		),
		top=rx.cond(
			DashboardState.chat_panel_mode == "floating",
			DashboardState.chat_float_top, "auto",
		),
		left=rx.cond(
			DashboardState.chat_panel_mode == "floating",
			DashboardState.chat_float_left, "auto",
		),
		z_index=rx.cond(
			DashboardState.chat_panel_mode == "floating", "50", "1",
		),
	)


# \u2500\u2500\u2500 Sections \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _header_bar() -> rx.Component:
	"""Clean header: collapse chevron | bot icon + title | dock/float toolbar."""
	return rx.hstack(
		# collapse chevron
		rx.el.button(
			rx.cond(
				DashboardState.chat_panel_collapsed,
				rx.icon("chevron-right", size=16),
				rx.icon("chevron-left", size=16),
			),
			on_click=DashboardState.toggle_chat_panel,
			title="Collapse / expand",
			background="transparent",
			color=_T3,
			border="none",
			border_radius=_R_SM,
			padding="0.3rem",
			cursor="pointer",
			display="inline-flex",
			align_items="center",
			justify_content="center",
			transition=_EASE,
			flex_shrink="0",
			_hover={"color": _CYAN},
		),
		# title
		rx.cond(
			DashboardState.chat_panel_collapsed,
			rx.box(),
			rx.hstack(
				rx.icon("bot", size=16, color=_CYAN),
				rx.text(
					"Claude Chatbot", size="3", weight="bold",
					color=_T1, font_family=_FONT, white_space="nowrap",
				),
				spacing="2",
				align="center",
				flex="1",
				min_width="0",
			),
		),
		# toolbar
		rx.cond(
			DashboardState.chat_panel_collapsed,
			rx.box(),
			rx.hstack(
				_tb(
					"panel-left", DashboardState.move_chat_to_left,
					"Dock left", DashboardState.chat_panel_mode == "left",
				),
				_tb(
					"panel-right", DashboardState.move_chat_to_right,
					"Dock right", DashboardState.chat_panel_mode == "right",
				),
				_tb(
					"maximize-2", DashboardState.set_chat_floating,
					"Float", DashboardState.chat_panel_mode == "floating",
				),
				spacing="1",
				align="center",
			),
		),
		justify="between",
		align="center",
		width="100%",
		padding_bottom="0.4rem",
		border_bottom=f"1px solid {_BORDER}",
		flex_shrink="0",
	)


def _collapsed_strip() -> rx.Component:
	"""Thin vertical strip when panel is collapsed."""
	return rx.vstack(
		rx.icon("message-circle", size=20, color=_CYAN),
		rx.el.span(
			"Chat",
			style={
				"writing-mode": "vertical-rl",
				"text-orientation": "mixed",
				"color": _T4,
				"font-size": "11px",
				"font-family": _FONT,
				"letter-spacing": "0.05em",
				"margin-top": "0.5rem",
			},
		),
		spacing="2",
		align="center",
		padding_top="1rem",
		flex="1",
	)


def _float_controls() -> rx.Component:
	"""Nudge arrows for floating mode repositioning."""
	return rx.cond(
		DashboardState.chat_panel_mode == "floating",
		rx.hstack(
			rx.icon("grip-horizontal", size=12, color=_T4),
			_nudge_btn("arrow-left", -30, 0),
			_nudge_btn("arrow-up", 0, -30),
			_nudge_btn("arrow-down", 0, 30),
			_nudge_btn("arrow-right", 30, 0),
			spacing="1",
			align="center",
			justify="center",
			width="100%",
			padding_y="0.15rem",
			flex_shrink="0",
		),
		rx.box(),
	)


def _instruction_block() -> rx.Component:
	"""Styled instruction hint card."""
	return rx.box(
		rx.hstack(
			rx.icon("info", size=13, color=_T4, flex_shrink="0"),
			rx.text(
				"Ask about spend trends, risks, variances, or chart insights.",
				color=_T3, size="2", line_height="1.4", font_family=_FONT,
			),
			spacing="2",
			align="start",
		),
		background=_BG_INSTR,
		border=f"1px solid {_BORDER}",
		border_radius=_R_MD,
		padding="0.55rem 0.7rem",
		width="100%",
		flex_shrink="0",
	)


def _chat_history() -> rx.Component:
	"""Scrollable message container."""
	return rx.box(
		rx.foreach(DashboardState.chat_messages, _message_bubble),
		id="chat-scroll",
		width="100%",
		flex="1 1 0",
		min_height="0",
		overflow_y="auto",
		display="flex",
		flex_direction="column",
		gap="0.5rem",
		padding="0.25rem 0.15rem 0.5rem 0",
		**_SCROLLBAR,
	)


def _error_block() -> rx.Component:
	"""Inline error banner."""
	return rx.cond(
		DashboardState.chat_error != "",
		rx.box(
			rx.hstack(
				rx.icon("triangle-alert", size=14, color=_RED),
				rx.text(DashboardState.chat_error, color="#FCA5A5", size="2"),
				spacing="2",
				align="center",
			),
			background="#2A1520",
			padding="0.55rem 0.75rem",
			border_radius=_R_MD,
			border=f"1px solid {_RED}",
			width="100%",
			flex_shrink="0",
		),
		rx.box(),
	)


def _input_zone() -> rx.Component:
	"""Sticky footer: text input + Ask Claude / Save / Email buttons."""
	return rx.box(
		rx.el.form(
			rx.el.input(
				name="chat_input",
				placeholder="Ask Claude about this dashboard\u2026",
				background=_BG_INPUT,
				border=f"1px solid {_BORDER}",
				color=_T1,
				border_radius=_R_MD,
				padding="0.65rem 0.8rem",
				font_size="14px",
				line_height="1.4",
				height="44px",
				font_family=_FONT,
				width="100%",
				box_sizing="border-box",
				outline="none",
				transition=_EASE,
				auto_complete="off",
				_focus={
					"border": f"1px solid {_CYAN}",
					"box_shadow": "0 0 8px rgba(62,231,224,0.18)",
				},
			),
			rx.hstack(
				# Primary: Ask Claude
				rx.el.button(
					rx.cond(
						DashboardState.is_chat_loading,
						"Thinking\u2026", "Ask Claude",
					),
					type="submit",
					background=_CYAN,
					color=_BG,
					border="none",
					border_radius=_R_MD,
					padding="0.5rem 0",
					font_size="13px",
					font_weight="700",
					font_family=_FONT,
					cursor="pointer",
					flex="1",
					transition=_EASE,
					_hover={
						"transform": "translateY(-1px)",
						"box_shadow": "0 6px 20px rgba(62,231,224,0.35)",
					},
				),
				# Secondary: Save Response
				rx.el.button(
					"Save",
					type="button",
					on_click=DashboardState.save_last_response,
					background="transparent",
					color=_CYAN,
					border=f"1px solid {_BCYAN}",
					border_radius=_R_MD,
					padding="0.5rem 0",
					font_size="13px",
					font_weight="600",
					font_family=_FONT,
					cursor="pointer",
					flex="0.65",
					transition=_EASE,
					_hover={
						"background": "rgba(62,231,224,0.08)",
						"transform": "translateY(-1px)",
					},
				),
				# Tertiary: Email Report
				rx.el.button(
					"Email",
					type="button",
					on_click=DashboardState.send_email_report,
					background="transparent",
					color=_BLUE,
					border=f"1px solid {_BBLUE}",
					border_radius=_R_MD,
					padding="0.5rem 0",
					font_size="13px",
					font_weight="600",
					font_family=_FONT,
					cursor="pointer",
					flex="0.65",
					transition=_EASE,
					_hover={
						"background": "rgba(76,141,255,0.08)",
						"transform": "translateY(-1px)",
					},
				),
				width="100%",
				spacing="2",
				margin_top="0.4rem",
			),
			on_submit=DashboardState.handle_chat_submit,
			reset_on_submit=True,
			width="100%",
			display="flex",
			flex_direction="column",
		),
		width="100%",
		flex_shrink="0",
		padding_top="0.5rem",
		border_top=f"1px solid {_BORDER}",
	)


# \u2500\u2500\u2500 Message Bubble \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def _message_bubble(message: dict) -> rx.Component:
	"""Chat bubble: user right-aligned blue, Claude left-aligned dark."""
	is_user = message["role"] == "user"
	is_saved = message["saved"]
	return rx.box(
		# saved badge with divider
		rx.cond(
			is_saved,
			rx.hstack(
				rx.box(flex="1", height="1px", background=_BORDER),
				rx.el.span(
					"\u2713 Saved",
					style={
						"color": _CYAN,
						"background": "rgba(62,231,224,0.10)",
						"font-size": "10px",
						"padding": "0.1rem 0.45rem",
						"border-radius": _R_PILL,
						"font-weight": "600",
						"font-family": _FONT,
						"white-space": "nowrap",
					},
				),
				rx.box(flex="1", height="1px", background=_BORDER),
				align="center",
				width="100%",
				margin_bottom="0.2rem",
			),
			rx.box(),
		),
		# bubble content
		rx.box(
			# role label + bookmark (assistant only)
			rx.cond(
				is_user,
				rx.box(),
				rx.hstack(
					rx.hstack(
						rx.icon("bot", size=12, color=_CYAN),
						rx.text(
							"Claude", color=_CYAN, size="1",
							weight="bold", font_family=_FONT,
						),
						spacing="1",
						align="center",
					),
					rx.el.button(
						rx.cond(
							is_saved,
							rx.icon("bookmark-check", size=12),
							rx.icon("bookmark", size=12),
						),
						on_click=DashboardState.save_response(message["id"]),
						title="Save this response",
						background="transparent",
						color=rx.cond(is_saved, _CYAN, _T4),
						border="none",
						cursor="pointer",
						padding="0.15rem",
						border_radius="4px",
						display="inline-flex",
						align_items="center",
						transition=_EASE,
						_hover={"color": _CYAN},
					),
					justify="between",
					align="center",
					width="100%",
					margin_bottom="0.3rem",
				),
			),
			# message content
			rx.cond(
				is_user,
				rx.text(
					message["content"], color=_T1, size="3",
					white_space="pre-wrap", font_family=_FONT,
					line_height="1.5",
				),
				rx.markdown(
					message["content"], component_map=_MD_MAP,
					color=_T2, font_family=_FONT,
					line_height="1.6", width="100%",
				),
			),
			# timestamp
			rx.cond(
				message["timestamp"] != "",
				rx.text(
					message["timestamp"], color=_T4, size="1",
					font_family=_FONT,
					text_align=rx.cond(is_user, "right", "left"),
					margin_top="0.3rem",
				),
				rx.box(),
			),
			# bubble styling
			background=rx.cond(
				is_user, _BG_USER,
				rx.cond(is_saved, _BG_SAVED, _BG_ASST),
			),
			border=rx.cond(
				is_saved,
				f"1px solid {_BCYAN}",
				f"1px solid {_BORDER}",
			),
			border_radius=rx.cond(
				is_user,
				"14px 14px 4px 14px",
				"14px 14px 14px 4px",
			),
			padding="0.7rem 0.85rem",
			max_width="88%",
			box_shadow=rx.cond(is_saved, _GLOW_CYAN, "none"),
			transition=_EASE,
			_hover={
				"border_color": _BCYAN,
				"transform": "translateY(-1px)",
			},
		),
		# alignment
		display="flex",
		flex_direction="column",
		align_items=rx.cond(is_user, "flex-end", "flex-start"),
		width="100%",
	)
