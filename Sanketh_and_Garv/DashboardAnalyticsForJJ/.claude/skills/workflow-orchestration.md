# Skill: Workflow Orchestration

## Purpose
Use this skill whenever Claude designs or updates dashboard workflows, interaction handlers, or state synchronization.

## Primary Workflow Chain
- Chain nodes in this order: load_data to compute_metrics to generate_charts to update_state.
- load_data reads and validates both Excel files, normalizes fields, and returns structured tabular payloads.
- compute_metrics derives executive summary metrics and grouped datasets from the validated inputs.
- generate_charts converts grouped outputs into chart-ready structures and Plotly figures.
- update_state maps workflow outputs into Reflex state so the UI updates cleanly.
- When risk heatmap data is prepared, update_state must preserve the derived Likelihood and Impact metadata so explanations and tooltips remain consistent.
- compute_metrics must accept time_range and return range-filtered metrics and grouped data.

## Mode and Tab Orchestration
- Support explicit analytics mode tabs with deterministic state keys and transition events.
- Preserve current mode context across filter changes and chat interactions.
- Persist left navigation collapsed or expanded preference and restore it during app load.
- Include transitions and tooltip behavior expectations in UI-focused workflow metadata when needed.

## Chat Workflow Chain
- Chain chatbot nodes in this order: load_memory to chat_query to update_state.
- load_memory reads .claude/memory/chat_context.md and returns memory context as structured text.
- chat_query sends user message, chat history, and loaded memory to Claude and returns assistant response payload.
- update_state appends both user and assistant messages to chat state.
- save_response writes selected assistant responses to persistent chat memory.
- chat panel ui state changes must not disrupt chat history continuity or save_response accessibility.

## Email Report Workflow Chain
- Chain email report workflow with send_email_report and state update output handling.
- send_email_report reads today's computed analytics payload from workflow state.
- send_email_report formats structured email content and dispatches email_tool.send_report.
- send_email_report must not include saved chat memory content.

## Agentic Triggers
- On click Risk Owner, call the email tool.
- On hover Variance, call the explain_variance node.
- On click Export, call the export_report node.
- Export actions should produce a PDF summary of the current filtered dashboard state.
- Hover-triggered explain_variance responses must be exactly two sentences.
- Export actions should write files to the project Reports directory.
- On click Save Response in chat, call save_response and append to persistent memory.
- On click Send Email Report, call send_email_report and open default mail client with pre-filled report.
- On hover chart elements, trigger deeper insight payload retrieval when available.

## Structured Outputs For Reflex
- Return dictionaries that contain status, metrics, charts, tables, actions, and errors when present.
- Keep outputs serializable and predictable so Reflex can consume them directly.
- Separate user-facing messages from machine-facing payload details.
- Include mode-aware payload sections for tabs, markers, and hover-depth content.
- Include chat_panel_state payload with mode, collapsed status, and optional floating coordinates.
- Include chart_expand payloads with selected chart id, description text, and additional insights.
- Include selected time_range in dashboard state outputs.
- Include active_filter_chip payloads for compact filter display.

## Orchestration Standards
- Keep side effects out of the main metrics-and-chart path unless the interaction explicitly requests them.
- Preserve current filter context in every workflow call.
- Return partial success information when one action fails but the rest of the dashboard payload remains valid.
- Keep transitions smooth by preferring incremental state updates over full-state resets.
- Keep chart expansion and close transitions smooth and non-blocking.
- Keep filter show or hide and advanced filter transitions smooth and non-blocking.

## Comprehensive UI and UX Remediation Requirements
- Left navigation must use full labels with icons, clear active highlighting, collapse and expand behavior, collapsed hover tooltips, grouped spacing, section dividers, and Deep Space Navy plus Electric Cyan futuristic styling.
- Filters must render as a compact horizontal bar with Advanced Filters expand and collapse, active filter chips, grouped spacing, neon-accented toggles, and reduced visual dominance.
- Chat panel must be collapsible, support dock left and dock right and floating mode, and animate transitions smoothly.
- Chat input must remain visible at all times with larger size, stronger contrast, clear placeholder text, and a neon-accented Ask Claude action.
- Chat history must use spaced bubble styling, persistent scroll container behavior, and Save Response controls on assistant messages.
- KPI cards must use meaningful titles, icon markers, trend indicators, consistent neon-accented card styling, and clean grid alignment.
- Charts must support hover tooltips, legends, axis labels, smooth transitions, styled containers, responsive multi-column grids, and cyan-blue-purple palette consistency.
- Dashboard spacing system must follow 8px and 16px and 24px rhythm, section headers, and clear visual dividers for hierarchy.
- Time range UX must use Today and Week and Month pill toggles with active highlighting and full analytics recalculation for metrics and charts, with subtitle feedback.
- Every chart must provide Expand action opening a modal with full chart, full legend, full axis labels, additional insights, close control, and smooth motion.
- Predictive insight card must use icon, title, neon border, generous spacing, hover animation, and placement near relevant charts.
