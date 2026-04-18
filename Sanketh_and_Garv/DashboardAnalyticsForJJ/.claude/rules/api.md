# API Rules

## External Integration Policy
- Use the Anthropic API for LLM calls.
- Keep the Anthropic client isolated in the agent layer.
- Validate request payloads before sending prompts and validate response structure before consuming model output.

## Interface Design
- Keep API clients thin and reusable.
- Avoid leaking transport details into business workflows.
- Convert external request and response payloads into internal structured dictionaries at the boundary.
- Keep any chart insight payloads used in expanded views compact and structured.
- If time-range state is transmitted through APIs in future, preserve explicit Today, Week, Month semantics.
- If filter summaries are transmitted, keep chip labels concise and deterministic.

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

## Operational Discipline
- Handle authentication through configuration rather than inline values.
- Surface timeout, rate-limit, and response errors as structured workflow errors.
- Keep external calls asynchronous or background-safe so the UI thread is never blocked.
- If persisted UI preferences are saved via API or backend endpoints in future, keep the interface deterministic and non-blocking.
- If floating chat drag metadata is persisted, store deterministic x and y coordinate values plus mode safely.
