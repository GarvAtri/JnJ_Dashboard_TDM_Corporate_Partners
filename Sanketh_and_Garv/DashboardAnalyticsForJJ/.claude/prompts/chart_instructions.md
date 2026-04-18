# Chart Instructions

## Plotly Styling Rules
- Use Plotly figures that prioritize executive readability over decorative complexity.
- Keep titles direct and business-oriented.
- Use clear axis labels derived from canonical analytics field names.
- Avoid chart clutter by limiting legends, annotations, and secondary encodings to what is necessary.
- Use smooth curves, rounded visual treatment, and readable spacing for enterprise dashboards.
- Favor modular chart cards with consistent margins and shadow depth.
- Keep chart cards minimal: title plus chart content with no clutter.
- Use consistent rounded card corners and consistent internal padding.

## Color Palette
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
- Gradients:
- Cyan #3EE7E0 to #1FB7B0
- Blue #4C8DFF to #2A6BE0
- Purple #A66BFF to #7A3CE0
- Keep categorical color usage consistent so status meanings remain stable across views.

## Layout Spacing
- Use consistent spacing around chart titles, legends, and plotting areas.
- Keep charts aligned on a clean dashboard grid.
- Prefer generous margins so filters, legends, and tooltips do not overlap or crowd the view.
- Maintain high whitespace ratio consistent with light enterprise dashboard composition patterns.
- Use thin grid lines and subtle glows for futuristic depth without clutter.
- Chart cards must reflow cleanly when navigation width animates between expanded and collapsed states.
- Place charts in a responsive two-column or three-column grid.
- Maintain consistent card heights and avoid overflow clipping.
- Ensure spacing remains clean when compact filter bar expands into advanced mode.

## Tooltip Formatting
- Format currency values as readable monetary amounts.
- Format percentages with a clear percent symbol and sensible precision.
- Format dates in a consistent business-friendly style.
- Include the grouped dimension, metric label, and relevant filter context when it changes interpretation.
- Include deeper insight snippets on hover when available from workflow payloads.

## Delivery Rules
- Every chart must be responsive.
- Every chart must derive from validated and filtered data.
- Every chart must expose enough metadata for Reflex rendering and Claude explanation.
- Support line, donut, and bar chart families with style coherence.
- Include trend markers and predictive cues when deterministic criteria are met.
- Preserve chart readability in responsive small-screen layouts with collapsed navigation.
- Preserve chart readability when chat is docked right or floating over content.
- Add Expand or View Large control for each chart card.
- Expanded chart mode must include full legend, full axis labels, full description, and additional insights.
- All chart views must reflect selected time range: Today, This Week, or This Month.

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
