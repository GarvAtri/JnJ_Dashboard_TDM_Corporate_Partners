# System Architecture

This document describes the overall architecture of the Dashboard Analytics platform — how data flows from Excel workbooks through analytics pipelines to interactive charts and AI-powered insights.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Reflex Web App                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         UI Layer (src/ui/)                           │   │
│  │  ┌────────────┐  ┌─────────────┐  ┌───────────┐  ┌──────────────┐  │   │
│  │  │  layout.py  │  │components.py│  │ charts.py │  │chat_panel.py │  │   │
│  │  │  Page comp. │  │  Reusable   │  │  Plotly   │  │  Claude chat │  │   │
│  │  │  + routing  │  │  widgets    │  │  builders │  │  sidebar     │  │   │
│  │  └──────┬─────┘  └──────┬──────┘  └─────┬─────┘  └──────┬───────┘  │   │
│  └─────────┼───────────────┼────────────────┼───────────────┼──────────┘   │
│            └───────────────┴────────────────┴───────────────┘              │
│                                     │                                      │
│                            ┌────────▼────────┐                             │
│                            │    state.py      │                             │
│                            │  DashboardState  │                             │
│                            │ (Reflex reactive │                             │
│                            │   state store)   │                             │
│                            └────────┬─────────┘                            │
│                                     │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │                     Agent Layer (src/agent/)                        │   │
│  │  ┌──────────┐  ┌────────────┐  ┌───────┐  ┌────────┐  ┌────────┐  │   │
│  │  │ graph.py │  │workflow.py │  │llm.py │  │chat.py │  │tools.py│  │   │
│  │  │ LangGraph│  │  Node      │  │Claude │  │  Chat  │  │ PDF,   │  │   │
│  │  │ compile  │  │  impls     │  │  API  │  │ nodes  │  │ Email  │  │   │
│  │  └──────────┘  └────────────┘  └───────┘  └────────┘  └────────┘  │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                     │                                      │
│  ┌──────────────────────────────────▼──────────────────────────────────┐   │
│  │                        Data Layer                                   │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────────────┐  │   │
│  │  │ Enterprise_Spend.xlsx   │  │ RiskRegisterSample.xlsx         │  │   │
│  │  │ PO_Header +             │  │ Risk IDs, owners, status,      │  │   │
│  │  │ SpendDetails_JobAid     │  │ severity, dates                 │  │   │
│  │  └─────────────────────────┘  └─────────────────────────────────┘  │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │ User Uploads (src/uploads/)                                 │   │   │
│  │  │ .xlsx, .xls, .json, .pdf → scrape → Claude column analysis  │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI Framework** | Reflex 0.8.28+ | Python-based reactive web framework |
| **Charts** | Plotly 6.6 | Interactive data visualizations |
| **AI Orchestration** | LangGraph 1.1.6 | Graph-based deterministic workflow engine |
| **LLM** | Anthropic Claude 3.5 Sonnet | Variance explanations and chat |
| **Data** | Pandas 3.0 + OpenPyXL | Excel ingestion and analytics |
| **PDF Export** | ReportLab 4.4 | Dashboard report generation |
| **Config** | python-dotenv | Environment variable management |
| **Testing** | pytest 9.0 | Unit and integration tests |

---

## Module Responsibilities

### UI Layer (`src/ui/`)

| Module | Responsibility |
|---|---|
| `layout.py` | Page composition, navigation rail, title bar, section routing |
| `components.py` | Reusable widgets: metric cards, chart cards, filter controls, file upload modal, column relevance report |
| `charts.py` | Plotly chart builders (treemap, bar, line, heatmap, histogram) |
| `chat_panel.py` | Claude chat sidebar with dock/float modes and message history |

### State (`src/state.py`)

Single `DashboardState` class — the reactive store that connects UI to workflows:

- Holds all UI state (filters, metrics, chart data, chat messages, preferences)
- Manages file upload lifecycle (upload, scrape, AI analysis, removal)
- Exposes event handlers that trigger agent workflows
- Persists user preferences to `.claude/memory/ui_state.json`

### Agent Layer (`src/agent/`)

| Module | Responsibility |
|---|---|
| `graph.py` | Compiles LangGraph workflows and provides `run_*_workflow()` entry points |
| `workflow.py` | Node implementations: data loading, metric computation, chart generation |
| `llm.py` | Anthropic API clients for variance explanations and chat |
| `chat.py` | Chat-specific nodes: memory loading, query, state update |
| `memory.py` | File-based persistent memory for chat context |
| `tools.py` | Side-effect tools: PDF export, mailto links |
| `email.py` | Daily analytics report formatting |
| `file_scraper.py` | Multi-format file metadata extraction (Excel sheets/columns/dtypes, JSON structure/keys, PDF headings/tables) |
| `column_analyzer.py` | Claude-powered column relevance classifier — maps uploaded columns to dashboard schema |

> See [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md) for detailed AI workflow diagrams and design decisions.

---

## Data Pipeline

### Input Sources

| File | Sheets | Key Fields |
|---|---|---|
| `Enterprise_Spend_Jan_2026.xlsx` | PO_Header, SpendDetails_JobAid | PO_Number, PO_Status, Business_Sector, Addressable_Flag, PO_Total_Amount, Variance_vs_Budget, Root_Cause_Code |
| `RiskRegisterSample.xlsx` | Single sheet | Risk #, Risk Owner, Risk Status, Risk Level, Risk Category, Open Date, Closed Date |

### Normalization & Derived Fields

1. Column name normalization (e.g., `Business_Sector` → `Sector`)
2. Timestamp coercion and flag standardization
3. Derived metrics:
   - `Days_Open` = current date − Open Date
   - `Impact_Score` = Low→2, Medium→3, High→5
   - `Likelihood_Score` = based on Days_Open buckets (1–5)

### KPI Metrics

| Metric | Computation |
|---|---|
| Total PO Volume | Count of unique `PO_Number` in filtered data |
| Average Variance | Mean of `Variance_vs_Budget` in filtered data |
| Active Risk Count | Risks where status ≠ Closed or Closed_Date is empty |
| Addressable Spend % | (Addressable spend / Total spend) × 100 |

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Left Nav (70px collapsed │ 260px expanded)  │  Chat Panel   │
├─────────────────────────────────────────────┤  (340px,      │
│                                             │   dock L/R    │
│  Title Bar                                  │   or float)   │
│  ───────────────────────────────────────    │               │
│  Filters (compact bar, expandable)          │  Message      │
│  Analytics Mode Tabs                        │  History      │
│  ───────────────────────────────────────    │  (scrollable) │
│  KPI Metric Cards (4-up grid)               │               │
│  ───────────────────────────────────────    │               │
│  Charts Grid (2–3 columns, responsive)      │  Input Zone   │
│  ───────────────────────────────────────    │  (always      │
│  Risk Owner Action Cards                    │   visible)    │
│                                             │               │
└─────────────────────────────────────────────┴───────────────┘
```

### Dashboard Modes

| Mode | Content |
|---|---|
| **Overview** | Combined spend + risk summary |
| **Spend** | Sector treemap, root-cause variance, trend/seasonality |
| **Risk** | Risk heatmap, aging histogram, risk owner cards |
| **Forecast** | Predictive insights and variance drift analysis |

### Charts

1. **Sector Spend Donut** — PO_Total_Amount by Sector, colored by PO_Status
2. **Root Cause Variance Bar** — Grouped horizontal bars by Root_Cause × Sector
3. **Trend & Seasonality Line** — Dual-axis: Monthly Spend + Cumulative Variance with forecast marker
4. **Risk Heatmap** — 5×5 Likelihood vs Impact matrix
5. **Aging Risk Histogram** — Days_Open distribution (12 bins)

### Design System

- **Theme**: Dark mode with neon accents
- **Primary BG**: Deep Space Navy `#0B1221`
- **Primary Accent**: Electric Cyan `#3EE7E0`
- **Secondary Accent**: Azure Pulse Blue `#4C8DFF`
- **Tertiary Accent**: Purple `#A66BFF`
- **Spacing**: 8/16/24px rhythm
- **Card Radius**: 12px with soft glow shadows
- **Transitions**: 200–300ms ease-in-out

---

## External Integrations

| Integration | Method | Purpose |
|---|---|---|
| Anthropic API | REST via `anthropic` SDK | Variance explanations, chat Q&A, column relevance analysis |
| Email Client | `webbrowser.open(mailto:...)` | Risk owner follow-up, daily report |
| File System | Direct read/write | Excel ingestion, PDF export, memory persistence, file uploads |
