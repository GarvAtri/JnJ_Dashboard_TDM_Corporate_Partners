# LangGraph Rules

## Node Design
- Each workflow node must do one thing.
- The primary node set is load_data, compute_metrics, generate_charts, update_state, explain_variance, email_risk_owner, export_report, send_email_report, load_memory, chat_query, and save_response.
- Keep each node focused on one transformation, decision, or side effect boundary.
- Include explicit nodes or node outputs for active tab mode and hover-insight payload generation when UX depends on them.
- Ensure compute_metrics explicitly consumes time_range from workflow filters.

## Data Contracts
- Nodes must return structured dicts.
- Every node output must include enough context for the next node or for Reflex state consumption.
- No global state.
- Include style-relevant metadata in outputs when UI needs deterministic rendering behavior.
- Include persisted UI preference payload keys when nav state must survive refresh cycles.
- Include persisted chat panel mode and collapse keys when chat layout depends on state restoration.
- Include chart metadata payloads for legend mode, axis label completeness, and expanded-view descriptions.
- Include selected time range in state payloads consumed by UI.
- Include active-filter summaries so UI chips can render deterministic labels.

## Side Effects
- Use tools for side effects such as email and file export.
- Keep explain_variance separate from data loading and chart generation so interpretation remains independently testable.
- Use save_response as the only workflow node that writes chat memory to disk.
- Keep chat_query pure except for LLM calls and structured response generation.
- Use send_email_report as the dedicated node for daily analytics email dispatch.
- send_email_report must not read from chat memory and must use analytics state payloads only.
- Keep UI mode transitions deterministic through workflow-driven state updates.

## Reliability
- Use explicit input and output keys for every node.
- Represent failures as structured error payloads rather than implicit exceptions in the UI path.
- Keep graph behavior observable through status and metadata fields.
- Ensure predictive insight markers are produced from deterministic calculations, not random heuristics.
- Ensure mode navigation state transitions and collapse-state restoration are deterministic.

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
