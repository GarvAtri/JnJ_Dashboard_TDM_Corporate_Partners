# Skill: Data Analysis

## Purpose
Use this skill whenever Claude generates or revises analytics logic for the dashboard. The objective is to turn the two Excel inputs into consistent executive metrics and grouped datasets for visualization and action.

## UX-Oriented Output Design
- Provide outputs that map cleanly into label then number then sublabel KPI hierarchy.
- Ensure BAN-ready metrics are emitted with display-friendly labels and concise supporting context.
- Emit tab-mode compatible payloads so users can switch analytics modes without recomputation side effects.
- Keep payload outputs lightweight so transitions remain smooth when layout width changes from nav collapse.
- Keep payload outputs lightweight so chat dock and floating transitions remain smooth.

## Data Preparation Rules
- Load spend and risk workbooks with pandas.
- Normalize spend sheet columns so Sector is a single canonical field and Root_Cause represents the variance root cause dimension.
- Normalize risk identifiers so Risk # becomes Risk_ID in the analytics model.
- Apply user-selected filters before computing metrics, charts, exports, or explanations.
- Apply selected time_range before computing metrics and grouped chart datasets.

## Executive Metric Definitions
- Total PO Volume: count unique PO_Number values in the filtered spend dataset.
- Average Variance: average Variance_vs_Budget in the filtered spend detail dataset.
- Addressable Spend %: sum of PO_Total_Amount where Addressable_Flag indicates addressable spend divided by total PO_Total_Amount in the filtered spend dataset.
- Active Risk Count: count risks whose status is not closed, or whose Closed Date is empty.

## Time-Series Calculations
- Monthly Spend: group spend by calendar month using the spend timestamp and sum PO_Total_Amount.
- Cumulative Variance: sort by timestamp and compute the running total of Variance_vs_Budget.

## Grouping Rules
- Group by Sector for sector-level spend analysis.
- Group by Root_Cause for variance attribution.
- Group by Status for operational monitoring and dashboard filters.
- Keep PO header aggregations and spend detail aggregations separate when the source data differs.

## Risk Aging Logic
- Compute Days_Open as Closed Date minus Open Date for closed risks.
- Compute Days_Open as current analysis date minus Open Date for active risks.
- Use integer day values for aging analysis.

## Risk Heatmap Derivation
- Derive Impact from Risk Level using a deterministic ordinal mapping.
- Map Low to 2, Medium to 3, and High to 5 on the 5-point impact scale.
- Derive Likelihood from Days_Open using deterministic aging bands so older unresolved items are treated as more likely to persist.
- Use this Likelihood mapping: 0 to 30 days = 1, 31 to 90 = 2, 91 to 180 = 3, 181 to 365 = 4, greater than 365 = 5.
- Use the derived Likelihood and Impact fields for the risk heatmap and keep the derivation visible in chart metadata and tooltips.

## Drift And Concentration Detection
- Detect variance drift by comparing rolling average variance over time and flagging sustained directional change rather than single-point spikes.
- Detect concentration risk with the 80/20 rule by ranking sectors, purchase orders, or selected dimensions by spend and checking whether the top 20 percent account for at least 80 percent of total spend.

## Predictive and Marker Guidance
- Derive trend markers from rolling metrics or variance direction changes using deterministic rules.
- Include predictive insight fields only when confidence thresholds are met by observed data signals.

## Output Expectations
- Return metrics in structured dictionaries with explicit labels and values.
- Return grouped datasets in chart-ready tabular form.
- Return derived fields such as Days_Open and cumulative measures as named columns.
- Include optional hover-insight strings for deeper interaction reveals when requested by UI workflows.
- Include concise per-chart description and additional insight strings for expanded chart views.
- Include selected time range in summary payload context.
- Provide active filter summaries suitable for compact chip rendering.

## Daily Email Summary Guidance
- Build daily report summaries from today's workflow outputs only.
- Include concise sections for Executive Summary, Spend and Variance, and Risk and Governance.
- Include quantified insights with metric values and directional context where available.
- Exclude any saved chat-response memory content from report summaries.

## Chatbot Dataset Guidance
- Chat responses should reference filtered dataset context whenever filters are active.
- For follow-up questions, prioritize continuity with the immediate chat history before recomputing broad summaries.
- When asked for chart explanations, map each explanation to chart dimensions, measures, and current filter scope.
- For root-cause questions, tie conclusions to grouped variance outputs and concentration risk findings.
- For risk and spend questions, report both current values and trend or aging context when available.

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
