# AI Architecture

This document describes the AI subsystem powering the Dashboard Analytics platform — from LLM integration and agent orchestration to persistent memory and fallback strategies.

---

## Overview

The AI layer provides two core capabilities:

1. **Variance Explanations** — CEO-friendly, 2-sentence root-cause analysis triggered on chart hover.
2. **Interactive Chat** — Multi-turn Q&A about spend trends, risk clusters, and chart interpretation with persistent memory.
3. **Column Relevance Analysis** — AI-powered classification of uploaded file columns against the dashboard’s SAP-inspired schema.

All AI interactions are orchestrated through **LangGraph** graphs with deterministic node sequencing and explicit input/output contracts.

---

## LLM Provider

| Property | Value |
|---|---|
| Provider | Anthropic |
| Model | `claude-sonnet-4-6` (Claude Sonnet) |
| Temperature | `0` (deterministic) |
| Max Tokens (Explanations) | 220 |
| Max Tokens (Chat) | 500 |
| Max Tokens (Column Analysis) | 4096 |
| Fallback | Deterministic responses when API key is missing or unavailable |

### Fallback Strategy

When `ANTHROPIC_API_KEY` is set to `"YOUR_KEY_HERE"` or the API is unreachable, the system gracefully degrades:

- **Variance explanations** fall back to pre-computed insights derived from `variance_drift` and `concentration_risk` metrics.
- **Chat responses** return a context summary based on current dashboard state instead of API-generated answers.
- **Column analysis** falls back to deterministic keyword matching against the known dashboard schema (spend, risk, variance fields).

---

## Agent Module Structure

```
src/agent/
├── llm.py              # Anthropic API clients (ClaudeVarianceExplainer, ClaudeChatAssistant)
├── workflow.py         # LangGraph node implementations (data loading, metrics, charts, explanations)
├── graph.py            # Graph compilation and workflow runners
├── chat.py             # Chat-specific workflow nodes
├── memory.py           # Persistent chat memory (load, save, append)
├── email.py            # Email report formatting and dispatch
├── tools.py            # Side-effect tools (mailto, PDF export)
├── file_scraper.py     # Multi-format file metadata extraction (Excel, JSON, PDF)
└── column_analyzer.py  # Claude-powered column relevance classification
```

---

## Workflow Graphs

### 1. Dashboard Workflow (Main Pipeline)

The primary 4-node graph that powers the entire dashboard on load or filter change:

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  load_data   │───▶│ compute_metrics  │───▶│ generate_charts  │───▶│  update_state    │
└──────────────┘    └──────────────────┘    └──────────────────┘    └──────────────────┘
      │                     │                       │                       │
  Validate &           Filter &               Plotly specs            Reflex-ready
  normalize            aggregate              from grouped            state shape
  Excel data           → metrics              data
```

**Nodes:**
- `load_data_node()` — Reads and validates Excel workbooks, normalizes column names, coerces types, computes derived fields (`Days_Open`, `Impact_Score`, `Likelihood_Score`).
- `compute_metrics_node()` — Applies user filters (sector, PO status, addressable flag, risk status, time range), computes 4 KPIs and grouped datasets.
- `generate_charts_node()` — Converts grouped data into 5 Plotly chart specifications.
- `update_state_node()` — Translates workflow output into the Reflex `DashboardState` shape.

### 2. Variance Explanation Workflow

Single-node graph triggered when a user hovers over a root-cause variance bar:

```
┌──────────────────────┐
│ explain_variance     │
│                      │
│ Context:             │
│ - root_cause         │
│ - variance_amount    │
│ - variance_drift     │
│ - concentration_risk │
│                      │
│ Output:              │
│ - 2-sentence         │
│   explanation        │
└──────────────────────┘
```

The `ClaudeVarianceExplainer` enforces exactly 2 sentences via `_two_sentence_text()` post-processing.

### 3. Chat Workflow

3-node pipeline for interactive Q&A:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│ load_memory  │───▶│  chat_query  │───▶│ update_chat_state│
└──────────────┘    └──────────────┘    └──────────────────┘
      │                     │                     │
  Read saved           API call with          Format for
  responses            history + memory       UI display
  from disk            + data context
```

**Prompt Construction:**
1. System instructions (role, tone, scope)
2. Saved memory (from `.claude/memory/chat_context.md`)
3. Recent chat history (last 12 messages)
4. Current data context (active filters, metrics, chart summaries)
5. User query

### 4. File Upload & Column Analysis Pipeline

Multi-step pipeline for uploaded file analysis:

```
User uploads files ──▶ handle_upload()
                            │
                   Save to src/uploads/
                   Validate extensions (.xlsx, .xls, .json, .pdf)
                            │
              ┌─────────────▼─────────────┐
              │   scrape_uploaded_files()  │
              │                           │
              │  file_scraper.py:          │
              │  - Excel: sheets, columns, │
              │    dtypes, sample values   │
              │  - JSON: structure, keys,  │
              │    record counts           │
              │  - PDF: headings, tables,  │
              │    page count, word count  │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │ analyze_columns_with_claude│
              │                           │
              │  column_analyzer.py:       │
              │  - System prompt with      │
              │    dashboard schema        │
              │  - Classify each column:   │
              │    spend | variance | risk │
              │    time | identifier |     │
              │    metadata | irrelevant   │
              │  - Confidence + reasoning  │
              │  - Schema mapping          │
              │  - Recommendation          │
              └─────────────┬─────────────┘
                            │
              Column Relevance Report in UI
              (color-coded, with stats)
```

**Known Dashboard Schema** (used for classification context):
- Spend Header: PO_Number, PO_Status, Business_Sector, Addressable_Flag, PO_Total_Amount, Last_Updated_Timestamp
- Spend Detail: PO_Number, Sector, Spend_Amount, Variance_vs_Budget, Root_Cause_Code, Last_Updated_Timestamp
- Risk Register: Risk #, Risk Description, Risk Owner, Risk Status, Risk Category, Risk Level, Risk ERM Type, Open Date, Closed Date

### 5. Supporting Workflows

| Workflow | Nodes | Purpose |
|---|---|---|
| Email Risk Owner | `email_risk_owner_node` | Opens `mailto:` draft pre-filled with risk status request |
| Export Report | `export_report_node` | Generates PDF summary via ReportLab |
| Send Email Report | `send_email_report_node` | Formats daily analytics summary and opens email client |

---

## Memory System

### Chat Memory

Persistent file-based storage at `.claude/memory/chat_context.md`:

- **`load_chat_memory()`** — Reads the memory file (creates it if missing).
- **`append_saved_response()`** — Appends a timestamped entry when the user clicks "Save Response" on an assistant message.
- **Injection** — Saved responses are automatically prepended to the chat prompt context, giving Claude continuity across sessions.

### UI State Persistence

User preferences are stored in `.claude/memory/ui_state.json`:

- Navigation collapsed/expanded state
- Chat panel mode (dock left / dock right / floating) and position
- Selected time range
- Filter visibility

---

## Data Flow: Variance Explanation (End-to-End)

```
User hovers over bar ──▶ explain_variance_from_hover()
                              │
                        Extract root_cause from hover event
                              │
                        Run variance workflow
                              │
                     ┌────────┴────────┐
                     │  API Available?  │
                     └────────┬────────┘
                        Yes   │   No
                    ┌─────────┼─────────┐
              Claude API call │   Deterministic
              (temp=0, 220    │   fallback from
               max tokens)    │   drift + concentration
                    └─────────┼─────────┘
                              │
                    _two_sentence_text() enforcement
                              │
                    predictive_insight_card update
```

---

## Data Flow: Chat (End-to-End)

```
User types message ──▶ ask_claude()
                            │
                   Add user message to UI (optimistic)
                            │
                   Background thread:
                     ┌──────┴──────┐
                     │ load_memory │  ← .claude/memory/chat_context.md
                     └──────┬──────┘
                     ┌──────┴──────┐
                     │ chat_query  │  ← Claude API (history + memory + data context)
                     └──────┬──────┘
                     ┌──────┴──────┐
                     │ update_state│  ← Format response for UI
                     └──────┬──────┘
                            │
                   Append assistant bubble to chat panel
                            │
              ┌─────────────┴─────────────┐
              │ User clicks "Save Response"│ (optional)
              └─────────────┬─────────────┘
                            │
              append_saved_response() ──▶ memory file
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Temperature 0 | Deterministic reasoning for consistent executive-facing outputs |
| 2-sentence limit on explanations | Forces concise, actionable root-cause summaries |
| File-based memory over database | Simplicity; single-user dashboard with no concurrency requirements |
| LangGraph over raw function calls | Explicit graph topology makes workflow dependencies visible and testable |
| Fallback mode | Dashboard remains fully functional without an API key for demos and development |
| Last 12 messages in chat context | Balances context window usage against conversation continuity |
