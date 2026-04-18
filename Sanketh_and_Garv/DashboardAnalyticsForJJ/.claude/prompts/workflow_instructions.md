# Workflow Instructions

## Node Design Principles
- Design each workflow node to perform one clear task and return a structured dictionary.
- Keep node inputs explicit and keep outputs small enough for downstream reasoning and UI updates.
- Make node behavior deterministic for the same validated input.
- Include mode-aware and interaction-aware fields needed by tabs and hover-depth UX.

## Required Node Definitions
- load_data: input is source configuration and current filter context; output includes normalized spend data, normalized risk data, validation status, and source metadata.
- compute_metrics: input is normalized data plus filter context; output includes executive metrics, grouped spend summaries, grouped variance summaries, grouped risk summaries, and derived fields such as Days_Open.
- generate_charts: input is metric-ready grouped data; output includes figure specifications for sector treemap, root cause variance bar chart, trend and seasonality line chart, risk heatmap, and aging risk histogram.
- update_state: input is the combined workflow payload; output includes the exact state shape needed by Reflex for metrics, charts, tables, and user-facing messages.
- explain_variance: input is selected variance context, grouped variance data, and current filter state; output includes explanation summary, supporting drivers, and confidence notes.
- email_risk_owner: input is selected risk context and owner details; output includes action status, recipient, subject, and body summary.
- export_report: input is current filter context plus visible metrics and chart-ready data; output includes export status, export metadata, and PDF file descriptor information.
- send_email_report: input is today's analytics metrics and insight payload; output includes report summary text, recipient, subject, mailto payload status, and dispatch metadata.
- load_memory: input is optional memory path; output includes persistent saved chat context.
- chat_query: input is user message, chat history, and saved memory; output includes assistant response, metadata, and updated conversation payload.
- save_response: input is selected assistant response and metadata; output includes persistence status and memory file location.
- ui_mode_switch: input is requested analytics mode; output includes active mode and mode-specific state payload.
- nav_collapse_toggle: input is requested collapse state; output includes persisted preference and responsive layout metadata.
- chat_panel_mode_switch: input is requested chat mode; output includes persisted mode and layout metadata.
- chat_panel_collapse_toggle: input is requested chat collapse state; output includes persisted state and active dimensions.
- chart_expand_open: input is selected chart id; output includes expanded chart metadata and insight payload.
- chart_expand_close: input is expanded chart state; output includes closed state confirmation.
- time_range_select: input is selected range value; output includes persisted range and dashboard refresh trigger.
- filters_toggle: input is requested show or hide state; output includes filter visibility and compact mode metadata.
- advanced_filters_toggle: input is requested expansion state; output includes expanded flag and transition metadata.
- hover_insight: input is chart interaction context; output includes deeper explanation payload and marker metadata.

## Reflex Output Schema
- Return top-level keys for status, metrics, charts, actions, data_summary, and errors.
- Use success and error payloads that Reflex can map directly into state.
- Include a user-facing message when an action completes or fails.
- For hover-based variance explanations, return exactly two sentences.
- For export actions, include a file path under the Reports directory.
- For chat_query actions, include assistant text, source model metadata, and serialized chat history.
- For save_response actions, include write status and updated memory summary.
- For send_email_report actions, include To, Subject, and dispatch status with mailto-encoded payload details.
- Include ui_mode payloads for tab navigation state.
- Include hover_detail payloads for glow-highlighted deeper-insight interactions.
- Include nav_state payloads with collapsed flag, width target, and persistence status.
- Include chat_panel_state payloads with dock position, floating mode flag, collapsed flag, and coordinate metadata.
- Include chart_expand_state payloads with chart id, expanded flag, legend mode, axis label mode, and insight text.
- Include time_range_state payload with selected mode and recomputation status.
- Include filters_state payload with visibility, expanded status, and active chips.

## Error Handling Patterns
- Fail fast on missing required columns or invalid workbook structures.
- Return structured errors rather than raising unhandled exceptions into the UI path.
- Separate recoverable validation issues from blocking execution failures.
- Allow unaffected dashboard sections to continue rendering when a non-critical action fails.
- send_email_report failures must return structured errors without breaking chat input or dashboard rendering.
- ui_mode_switch and hover_insight failures must degrade gracefully while preserving primary dashboard rendering.
- nav_collapse_toggle failures must degrade gracefully by keeping navigation usable in last known valid state.
- chat panel mode or collapse failures must degrade gracefully while preserving input visibility and saved-response access.

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
