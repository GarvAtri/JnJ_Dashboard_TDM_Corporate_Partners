# Skill: Email Actions

## Purpose
Use this skill whenever Claude generates the logic for contacting a risk owner from the dashboard.

## Action Pattern
- Use Python webbrowser to open mailto:.
- Resolve the recipient from the selected risk owner record.
- Build the email only after the user has selected a specific risk.

## Subject Format
- Status update requested for Risk ID {id}

## Body Requirements
- Auto-fill the body with a concise summary of the selected risk.
- Include the risk identifier, description, current status, owner, open date, days open, and requested follow-up.
- Keep the wording appropriate for operational follow-up and leadership visibility.

## Safety Rules
- Require a valid risk identifier and risk owner before preparing the action.
- Preserve current dashboard context when it materially affects the summary.
- Return a structured result that tells Reflex whether the mailto action was prepared successfully.
- Email actions must not interfere with navigation collapse animations or responsive layout behavior.
- Email actions must not interfere with chat panel docking, floating mode, or collapse animations.
- Email actions must not interfere with expanded chart modals or chart interaction state.
- Email and export actions should respect the currently selected time range context.
- Email and export actions should respect active filter selections even when filter bar is hidden.

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
