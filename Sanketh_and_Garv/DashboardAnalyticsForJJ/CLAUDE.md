# DashboardAnalyticsForJJ — Claude-powered Spend, Variance, and Risk Analytics Dashboard

## Analytics Vision
DashboardAnalyticsForJJ combines enterprise spend data and risk register data into one executive analysis surface. Claude is responsible for helping build a dashboard that explains spend behavior, budget variance, unresolved risk exposure, and the next actions leadership should take.

## UI and UX Direction
- The product visual language blends Nexo-style enterprise clarity with Rinesk-style futuristic analytics energy.
- Use a dark-mode base with high-contrast neon accents while preserving enterprise readability and generous whitespace.
- Prioritize modular cards, large KPI BANs, and smooth chart interactions over dense or cluttered screens.
- Use tabs to switch analytics modes and provide hover-driven detail reveals for deeper insight layers.

## High-Level Goals
- Load two Excel files: Enterprise_Spend_Jan_2026.xlsx and RiskRegisterSample.xlsx.
- Support multi-file upload of Excel, JSON, and PDF files with drag-and-drop interface.
- Scrape uploaded files to extract column names, data types, sample values, and structural metadata.
- Use Claude AI to classify uploaded columns as relevant or irrelevant against the SAP-inspired dashboard schema.
- Generate executive summary metrics: Total PO Volume, Average Variance, Active Risk Count, and Addressable Spend %.
- Generate spend and variance visualizations: Sector Treemap, Root Cause Variance Grouped Horizontal Bar Chart, and Trend & Seasonality Dual-Axis Line and Area Chart.
- Generate risk visualizations: Risk Heatmap and Aging Risk Histogram.
- Enable agentic actions: email risk owners, explain variance root causes, and export filtered reports.

## Layout and Component System
- Include a left vertical navigation rail for top-level navigation and mode access.
- The left navigation rail must be collapsible and animate between expanded and collapsed widths.
- Keep the Claude chat panel as a dedicated left-side workspace within the navigation experience.
- Use a modular card grid with 12px corner radius and soft glow shadows.
- Maintain a high whitespace ratio to support executive scanability.
- Include dropdown filters and widget-style CTA elements in a consistent control zone.
- Use large BAN KPI tiles with text hierarchy label to number to sublabel.

## Dashboard Title Requirements
- Page title must use: Dashboard Analytics — Spend | Risk | Forecast.
- Subtitle must show selected date range context: Today, This Week, or This Month.
- Title area must be visually separated from filters and charts.
- Typography for title must use Inter or SF Pro at 700 weight.
- Include Electric Cyan accent underline or bar in the title region.

## Filter UX Requirements
- Provide a compact filter bar with minimal vertical height.
- Provide an expandable Advanced Filters section for secondary controls.
- Advanced Filters expansion and collapse must animate smoothly.
- Show active filters as clear rounded chips with neon-accent styling.
- Provide a Show or Hide Filters control to collapse the filter bar entirely.
- Filter area must not clutter the dashboard content.
- Filter visuals must match futuristic styling with rounded chips and Electric Cyan accents.

## Mandatory Color System
- Primary:
- Deep Space Navy #0B1221
- Electric Cyan #3EE7E0
- Azure Pulse Blue #4C8DFF
- Secondary:
- Soft Graphite #1E2635
- Slate Grey #8A93A6
- Frost White #F5F7FA
- Semantic:
- Success Green #4CD964
- Warning Amber #FFC043
- Risk Red #FF5A5F
- Chart Gradients:
- Cyan #3EE7E0 to #1FB7B0
- Blue #4C8DFF to #2A6BE0
- Purple #A66BFF to #7A3CE0

## Typography and Motion
- Use Inter or SF Pro or Poppins across the dashboard and chat surfaces.
- BAN values should be rendered in the 48 to 64px range.
- Labels should use 12 to 14px sizing with clear hierarchy.
- Use smooth transitions for card hover, tab switches, and interactive controls.
- Navigation open and close transitions must use 200 to 300ms ease-in-out timing.
- Apply subtle glowing highlights and thin grid lines for futuristic emphasis without visual overload.

## Collapsible Navigation Requirements
- Provide clear affordance for open and close using a chevron, animated arrow, or hamburger button.
- Expanded and collapsed state must persist and be remembered across refresh and subsequent visits.
- Collapsed mode must show icon-first navigation and hide label text.
- When collapsed, hovering navigation icons must show tooltips for Overview, Spend, Risk, and Forecast.
- Navigation must be responsive on small screens with accessible toggle behavior and preserved usability.
- Navigation styling must remain consistent with Deep Space Navy background and Electric Cyan interactive accents.

## Dashboard Structure
- The sidebar is reserved for Agent Chat and dashboard actions initiated through Claude.
- The main area is reserved for dashboard visualizations, executive metrics, and filter-driven analysis.
- The dashboard flow should start with executive summary metrics, continue into spend and variance analysis, then move into risk analysis and action outputs.
- Shared filter state must drive all metrics, charts, drilldowns, explanations, and exports.

## Claude Chatbot Panel Requirements
- A dedicated left-side vertical Claude chatbot panel is mandatory.
- The chat panel must support collapse and expand with smooth animation.
- The chat panel must support dock left, dock right, and floating panel modes.
- Chat history must be scrollable and persist in state during the active session.
- A message input box must be fixed at the bottom of the panel.
- Every assistant message must include a Save Response action.
- A Send Email Report button must appear next to Ask Claude in the chat panel.
- Saved responses must be appended to a persistent memory file at .claude/memory/chat_context.md.
- Saved responses must be automatically injected into future Claude calls.
- Chat panel visual system must use Deep Space Navy #0B1221, Soft Graphite #1E2635, Electric Cyan #3EE7E0, 12px rounded corners, smooth hover animations, and Inter or SF Pro or Poppins typography.

## Chat Panel UX Requirements
- Input must always remain visible and never be clipped while scrolling messages.
- Buttons must be larger touch targets with neon-accented visual emphasis.
- Message bubbles must use improved padding, spacing, and readable bubble hierarchy.
- Typing visibility must be preserved at all times while the user enters text.
- Floating mode must behave as a draggable panel with visible drag affordance.
- Panel transitions for collapse and docking must use smooth 200 to 300ms ease-in-out timing.
- Chat panel must remember collapsed state and remembered position mode across reloads.
- Saved responses must remain accessible across all panel positions.

## Send Email Report Requirements
- Send Email Report must generate a concise summary of today's analytics only.
- The summary must include Executive Summary metrics, Spend and Variance insights, and Risk and Governance insights.
- The action must open the default email client via mailto with pre-filled To, Subject, and Body.
- To must use analytics-team@example.com as the default placeholder distribution list.
- Subject must follow Daily Analytics Report — {today's date}.
- Saved chat responses must not be included in Send Email Report content.

## Claude Chatbot UX Requirements
- Users can ask dataset questions and follow-up questions in natural language.
- Claude must answer with Anthropic API-backed reasoning.
- Responses must support data insights, chart explanations, root-cause analysis, risk analysis, and spend analysis.
- Chat requests must include user message, chat history, and saved memory context.

## Data Sources
- Enterprise_Spend_Jan_2026.xlsx is the spend workbook and contains purchase order and variance-related fields.
- RiskRegisterSample.xlsx is the risk workbook and contains risk ownership, status, severity, open and closed dates, and aging context.
- Claude should normalize spend sheet columns into a canonical analytics model when names differ between sheets. Treat Business_Sector and Sector as the same dimension. Treat Root_Cause_Code as the canonical root cause field for variance analysis.
- Because the risk workbook does not provide explicit Likelihood and Impact columns, Claude must derive them deterministically from available fields before generating the risk heatmap.

## Chart and Interaction Rules
- Required chart families include smooth line charts, donut charts, and bar charts with rounded visual treatment.
- Use neon-accented chart strokes and gradient fills while preserving readability on dark backgrounds.
- Apply hover interactions that reveal deeper analytics context and predictive markers when available.
- Surface trend markers and forward-looking cues in a clear, non-speculative manner tied to computed data.
- Use clean, minimal chart cards that contain title and chart content only.
- Axis labels must remain readable, aligned, and fully visible.
- Legends must be compact and positioned so they do not overlap plotted data.
- Use Electric Cyan, Azure Pulse Blue, and Purple gradient tones as primary chart accents.
- Grid lines must be thin and subtle.
- Data points should show neon glow emphasis on hover.
- Chart containers must use rounded corners with consistent padding and spacing.

## Chart Layout Requirements
- Charts must align in a consistent responsive grid using two-column or three-column arrangements.
- Chart cards must maintain consistent heights for visual rhythm.
- Charts must not overflow or clip within their cards.
- Every chart card must include an Expand or View Large action.
- Expanded chart view must include full legend visibility, full axis labels, full description text, and additional insights.

## Workflow Expectations
- Primary dashboard workflow: load_data, compute_metrics, generate_charts, update_state.
- Explanation workflow: explain_variance after a variance interaction.
- Communication workflow: email_risk_owner after a risk-owner action.
- Export workflow: export_report after an export request.
- Report email workflow: send_email_report after a Send Email Report action.
- Chat workflow: load_memory, chat_query, update_state.
- Save workflow: save_response for durable memory writes.
- Every workflow step must emit structured outputs that Reflex can consume directly.
- The update_state step is responsible for translating workflow outputs into the exact state shape used by Reflex components.
- Hover-triggered variance explanations must be exactly two sentences.
- Export actions must save PDF outputs under a Reports folder in the project root.
- Send Email Report must be instant and non-blocking from the user perspective.

## Time Range Requirements
- Analytics must support exactly three time modes: Today, This Week, and This Month.
- All executive metrics and all charts must refresh based on the selected time range.
- Time range selector must be compact, visually distinct, and placed in a primary controls area.
- Time range selector must use futuristic UI styling with pill buttons and neon glow emphasis.
- Selected time range must persist across dashboard navigation and mode changes.

## Time Range Backend Contract
- compute_metrics must accept and use a time_range parameter.
- Time filtering must occur before grouped metrics and chart datasets are calculated.
- Executive Summary payload must reflect the currently selected time range.

## Claude Rules
- Keep code modular.
- Use Reflex for UI.
- Use LangGraph for workflows.
- Use Anthropic API for LLM calls.
- Never block the UI thread.
- Always validate data before charting.
- Keep analytics logic separate from UI rendering.
- Keep side effects in explicit tool boundaries rather than inside metric or chart functions.
- Keep all UI outputs aligned with the defined design tokens and style guide.

## How To Run
Run the project with reflex run.

## Validation Expectations
- Validate source workbook structure before computing metrics.
- Validate filter application before generating charts or exports.
- Validate that every chart uses the intended dimensions and measures defined in the skills documentation.
- Validate that workflow outputs match the structured contracts defined in the workflow instructions.
- Validate all Claude-generated work with the test suite before finalizing changes, with emphasis on workflow sequencing, chart integrity, and metric correctness.
- Validate that every UI surface adheres to the mandatory color, typography, spacing, and interaction rules.

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
