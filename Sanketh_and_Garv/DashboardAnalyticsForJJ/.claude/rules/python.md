# Python Rules

## Core Standards
- Use type hints everywhere.
- Keep functions pure unless inside Reflex State.
- Keep modules focused on one responsibility.
- Document public functions, workflow nodes, and state transitions with concise docstrings.

## Design Token Standards
- Define and reuse centralized constants for color tokens, semantic colors, and chart gradients.
- Never duplicate hard-coded color literals across multiple modules when a shared token exists.
- Keep typography scale values explicit and reusable for BAN sizes and label sizes.
- Keep navigation width, transition timing, and collapse icon configuration in reusable constants.
- Keep chat panel mode keys, dimensions, and transition timing in reusable constants.
- Keep title bar and filter-bar styling constants centralized.

## Data Handling
- Use pandas for Excel ingestion.
- Validate required columns, value types, and missing data conditions before computation.
- Normalize workbook columns into a canonical analytics model before downstream use.
- Never hardcode file paths; use a config loader.
- Keep time-range filtering deterministic and centralized in analytics workflow helpers.

## Chatbot Backend Rules
- Load environment variables with load_dotenv and read the key using os.getenv("ANTHROPIC_API_KEY").
- Keep chat memory operations in a dedicated memory module.
- Persist saved assistant responses in .claude/memory/chat_context.md.
- Load persisted memory for each new Claude chat request.
- Ensure chat history and memory are passed as explicit structured inputs to chat workflows.

## Email Report Rules
- Implement an email tool with signature email_tool.send_report(to, subject, body).
- Build mailto URLs with URL-encoded subject and body.
- Open mailto links via webbrowser.open without blocking UI event loops.
- Use only today's analytics workflow outputs for report summaries.
- Exclude saved chat memory from report email generation.

## Analytics And Charting
- Use Plotly for charts.
- Keep data preparation separate from visual configuration.
- Ensure metrics and grouped outputs are returned as plain structured data.
- Support smooth line, donut, and bar chart outputs with style metadata.
- Use gradient-capable configuration paths for cyan, blue, and purple chart themes.
- Ensure predictive markers and trend annotations are deterministic and data-driven.
- Return chart metadata needed for expand-view descriptions and additional insights.
- Keep chart card sizing hints deterministic so UI can enforce consistent heights.

## Workflow Discipline
- Keep LangGraph nodes deterministic.
- Avoid hidden side effects in analytics functions and workflow nodes.
- Reserve side effects such as export and email launch for dedicated tool functions.
- Ensure hover-detail payloads and tab-mode payloads are explicit in returned workflow structures.
- Persist UI preference state for navigation collapse and restore it deterministically.
- Persist chat panel state including collapsed flag and dock mode.
- Persist selected time range state where UI persistence is expected.
- Preserve filter state values for compact chips and advanced filter expansion behaviors.

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
