# Skill: Chart Generation

## Purpose
Use this skill whenever Claude generates chart data mappings or Plotly configuration for the dashboard. Every chart must come from validated and filtered analytics outputs.

## Visual Style System
- Default chart surfaces should align with dark mode base using Deep Space Navy and Soft Graphite contexts.
- Use neon accents for active lines, highlights, and key comparative states.
- Apply thin grid lines and soft glow emphasis for interactive focus without reducing readability.
- Use gradient palettes consistently:
- Cyan gradient #3EE7E0 to #1FB7B0
- Blue gradient #4C8DFF to #2A6BE0
- Purple gradient #A66BFF to #7A3CE0
- Chart cards should remain minimal and clean with title plus chart content.
- Axis labels should be readable and aligned in both default and expanded chart views.
- Legends should be compact in default cards and fully visible in expanded views.
- Grid lines should remain thin and subtle.
- Data markers should include neon-glow hover emphasis.

## Sector-Wise Spend Treemap
- Area = PO_Total_Amount.
- Color = PO_Status.
- Group by Sector.
- Use the treemap to highlight concentration of spend by sector and status.

## Root Cause Variance Bar Chart
- X = Sum Variance_vs_Budget.
- Y = Root_Cause.
- Use grouped horizontal bars.
- Sort by materiality so the biggest variance drivers appear first.

## Trend And Seasonality Dual-Axis Line Chart
- X = Timestamp.
- Y1 = Monthly Spend.
- Y2 = Cumulative Variance.
- Aggregate to monthly grain and maintain chronological ordering.
- Render Monthly Spend on the primary axis and Cumulative Variance on the secondary axis.
- Use a line and area combination so trend and accumulation can be read together.
- Add trend markers where statistically meaningful drift is detected.

## Risk Heatmap
- Use a 5x5 grid with derived Likelihood on one axis and derived Impact on the other.
- Color scale: green to red.
- Metric: count of Risk_ID.
- Keep axis severity ordered from low to high.
- Derive Impact from Risk Level and derive Likelihood from Days_Open using the data-analysis skill rules.

## Aging Risk Histogram
- Data = Days_Open.
- Design the histogram to make long-tail risks obvious.
- Use bins that reveal both common aging ranges and extreme outliers.

## Donut and BAN Companion Guidance
- Use donut charts for high-level composition ratios when categorical share is important.
- Pair donut and bar visuals with BAN tiles to preserve label to number to sublabel hierarchy.

## General Rules
- Every chart must state its metric and grouping dimension clearly.
- Titles, legends, and labels must use executive language rather than raw system terms.
- Chart outputs must be compatible with Reflex rendering through Plotly figures.
- Rounded edges and smooth curves should be used where chart type supports them.
- Hover interactions must reveal deeper insight text, not just raw values.
- Chart containers must adapt responsively when the left navigation collapses or expands.
- Chart containers must also adapt when chat panel docks right or enters floating mode.
- Every chart card must expose an Expand or View Large action.
- Expanded chart mode must show full legend, full axis labels, full description, and additional insights.
- Every chart must regenerate from datasets filtered by selected time range.
- Charts should preserve legibility when filter bar is shown, hidden, expanded, or collapsed.

## Chatbot Chart Explanation Rules
- Chat explanations of charts must name the chart type and the mapped axes or fields.
- Explanations must include at least one material insight and one actionable interpretation.
- Follow-up chart questions should reuse saved chat memory context when available.

## Email Report Insight Rules
- Email report chart insights must summarize chart outcomes in plain language rather than raw traces.
- Include one concise spend and variance insight and one concise risk and governance insight.
- Keep report insight text suitable for a pre-filled email body.

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
