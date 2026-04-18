# Reflex Rules

## Component Rules
- All UI components must be pure functions.
- Keep analytics and workflow logic out of rendering functions.
- Use shared components for repeated metric, filter, and action patterns.
- Use modular card components with 12px radius and soft glow shadow treatments.

## State Rules
- State mutations only inside State class.
- Keep state transitions explicit and traceable.
- Maintain one source of truth for filters so all visualizations and actions stay aligned.
- Maintain one source of truth for active analytics tab or mode selection.
- Maintain one source of truth for selected analytics time range.

## Layout Rules
- Sidebar = Agent Chat.
- Main area = Dashboard visualizations.
- Use responsive layout with hstack and vstack.
- Place executive summary metrics above detailed visualizations.
- Chat panel must be a fixed-width left rail around 300px.
- Chat panel must support left dock, right dock, and floating mode.
- Chat history region must scroll independently from the dashboard area.
- Input area must remain anchored at the bottom of the chat panel.
- Add a left vertical navigation bar for global sections and analytics mode entry points.
- Left navigation must support collapse and expand interaction with animated width changes.
- Use a modular grid layout for cards, KPI BAN tiles, and chart panels.
- Preserve generous whitespace to avoid dense dashboard blocks.
- Keep dashboard content responsive as nav width changes.
- Keep time-range selector compact, distinct, and positioned for quick access.
- Add a dedicated title bar area separated from filter and chart content.
- Title area must include an Electric Cyan accent line or bar.

## Chat Panel Layout Rules
- Add a chat collapse and expand control with clear icon affordance.
- Add a Move to Right toggle and floating mode toggle.
- In floating mode, expose a visible drag handle and keep the panel above content layers.
- Keep input visible while chat history scrolls.
- Animate panel width and docking transitions using 200 to 300ms ease-in-out.
- Keep button hit areas large and readable in all panel modes.
- Ensure panel mode changes do not block chart rendering or state updates.

## Collapsible Navigation Rules
- Include an explicit toggle button with a recognizable affordance such as chevron, animated arrow, or hamburger icon.
- Use 200 to 300ms ease-in-out transitions for nav width and label visibility.
- Expanded width target should be around 240 to 260px and collapsed width around 64 to 72px.
- In collapsed mode, render icon-only entries and show hover tooltips for each destination.
- Persist collapsed or expanded preference and restore it on load.
- Ensure nav behavior remains usable on small screens through responsive width and toggle logic.
- Use Deep Space Navy #0B1221 for base surfaces and Electric Cyan #3EE7E0 for active and hover accents.

## Chatbot Styling Rules
- Use Deep Space Navy #0B1221 as the chat panel background.
- Use Soft Graphite #1E2635 for message cards.
- Use Electric Cyan #3EE7E0 for key accents, focus states, and action emphasis.
- Use 12px rounded corners for cards and input elements.
- Apply smooth hover transitions on interactive elements.
- Use Inter or SF Pro or Poppins for panel typography.
- Use soft glow emphasis for primary chat and CTA actions.
- Increase message bubble spacing and padding for readability.
- Keep futuristic glow accents and rounded corners in all panel modes.

## Chatbot Interaction Rules
- Assistant messages must include a Save Response button.
- Save Response actions must call a state event that persists memory.
- Chat panel actions must not block dashboard rendering.
- Send Email Report must appear beside Ask Claude in the chat input action row.
- Send Email Report must use Electric Cyan accent styling, 12px corners, and hover glow.
- Send Email Report action must dispatch a background-safe state event.

## Chart Rules
- Use rx.plotly() for charts.
- Treat Plotly figures as structured outputs passed into the UI.
- Keep charts responsive for executive dashboard use across common screen sizes.
- Support smooth line charts, donut charts, and bar charts with rounded visual style.
- Use neon accents, thin grid lines, and subtle glow highlights for futuristic contrast.
- Use hover states to reveal deeper insight text or metric overlays.
- Render charts in minimal cards with title and chart as the primary structure.
- Keep legends compact and prevent overlap with chart drawing area.
- Maintain readable axis labels with consistent alignment and spacing.
- Add per-chart Expand or View Large actions.
- Expanded chart view must include full legend, full axis labels, description, and additional insights.

## Chart Layout Rules
- Use responsive chart grids with two or three columns depending on available width.
- Keep chart card heights consistent to avoid jagged rows.
- Prevent clipping and overflow of chart canvases and labels.
- Enforce rounded corners and consistent internal padding across all chart cards.
- Ensure chart views react immediately when time range changes.

## Time Range UI Rules
- Implement three pill buttons for Today, This Week, and This Month.
- Apply neon glow on active selection and maintain readability in inactive states.
- Trigger dashboard recomputation when time range selection changes.
- Preserve selected time range across mode and navigation transitions.

## Filter Bar Rules
- Implement compact filter bar as default state.
- Add Show or Hide Filters toggle that can collapse filter region.
- Add Advanced Filters collapse and expand container with smooth animation.
- Show active filter chips in a single compact row with wrapping support.
- Keep filter controls visually lightweight and non-cluttering.

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

## Typography Rules
- BAN values must render in 48 to 64px range.
- Labels and sublabels should use 12 to 14px sizing with clear contrast.
- Follow hierarchy order label then number then sublabel in KPI components.
