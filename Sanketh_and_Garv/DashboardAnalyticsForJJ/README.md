# DashboardAnalyticsForJJ

A Claude-powered **Spend, Variance, and Risk Analytics Dashboard** that combines enterprise spending data and risk register data into a single executive analysis workspace. Built with **Reflex** (UI), **LangGraph** (orchestration), and **Anthropic Claude** (AI reasoning).

---

## Architecture

| Document | Description |
|---|---|
| [System Architecture](docs/ARCHITECTURE.md) | Overall system design — tech stack, data pipeline, UI layout, module responsibilities |
| [AI Architecture](docs/AI_ARCHITECTURE.md) | AI subsystem deep dive — LLM integration, LangGraph workflows, memory system, fallback strategy |

---

## Features

### Executive Analytics
- **4 KPI Metric Cards** — Total PO Volume, Average Variance, Active Risk Count, Addressable Spend %
- **Time Range Modes** — Today, This Week, This Month with full metric/chart recalculation
- **Multi-filter System** — Sector, PO Status, Addressable Flag, Risk Status with active filter chips
- **4 Dashboard Views** — Overview, Spend, Risk, Forecast modes via tabbed navigation

### Interactive Charts (Plotly)
- **Sector Spend Donut** — PO amounts by sector, colored by PO status
- **Root Cause Variance Bar** — Grouped horizontal bars by root cause × sector
- **Trend & Seasonality Line** — Dual-axis monthly spend vs cumulative variance with forecast marker
- **Risk Heatmap** — 5×5 likelihood vs impact severity matrix
- **Aging Risk Histogram** — Days-open distribution highlighting long-tail unresolved risks
- **Expand-to-Modal** — Every chart supports full-screen view with legends, axis labels, and insights

### Multi-File Upload & AI Analysis
- **Multi-File Upload** — Drag & drop or browse to upload Excel (.xlsx/.xls), JSON, and PDF files
- **File Scraper** — Automatic extraction of column names, data types, sample values, sheet structure, and PDF headings
- **Claude Column Analyzer** — AI-powered classification of every column as spend, variance, risk, time, identifier, metadata, or irrelevant
- **Relevance Report** — Color-coded report with confidence levels, schema mapping, and actionable recommendations
- **Deterministic Fallback** — Keyword-based column matching when Claude API is unavailable

### AI-Powered Insights (Claude Sonnet)
- **Variance Explanations** — Hover over a root-cause bar to get a 2-sentence CEO-friendly root-cause analysis
- **Interactive Chat** — Ask Claude about spend trends, risk clusters, and chart interpretation
- **Persistent Memory** — Save chat responses for continuity across sessions
- **Graceful Fallback** — Dashboard remains fully functional without an API key

### Agentic Actions
- **Email Risk Owner** — One-click `mailto:` draft pre-filled with risk status request
- **Export PDF Report** — Generate a dashboard summary PDF with metrics and insights
- **Send Daily Report** — Compose and dispatch a concise daily analytics email

### UI/UX
- **Dark Mode Design** — Deep Space Navy base with Electric Cyan / Azure Blue / Purple accents
- **Collapsible Navigation** — Left rail with icon-only collapsed mode and hover tooltips
- **Flexible Chat Panel** — Dock left, dock right, or floating draggable mode
- **State Persistence** — Nav, chat, and time range preferences saved across reloads
- **Responsive Layout** — 2–3 column chart grid with consistent spacing

---

## Project Structure

```
src/
├── app.py              # Reflex app entry point
├── state.py            # Reactive state store (DashboardState)
├── agent/
│   ├── graph.py        # LangGraph workflow compilation
│   ├── workflow.py     # Node implementations (data, metrics, charts)
│   ├── llm.py          # Anthropic API clients
│   ├── chat.py         # Chat workflow nodes
│   ├── memory.py       # Persistent chat memory
│   ├── tools.py        # PDF export, mailto tools
│   ├── email.py        # Daily report formatting
│   ├── file_scraper.py # Multi-format file metadata extraction
│   └── column_analyzer.py # Claude-powered column relevance classifier
└── ui/
    ├── layout.py       # Page composition and routing
    ├── components.py   # Reusable widgets (cards, filters, modals)
    ├── charts.py       # Plotly chart builders
    └── chat_panel.py   # Claude chat sidebar
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/) (optional — dashboard works without it)

### Installation

```bash
# Clone the repository
git clone https://github.com/sankygit/DashboardAnalytics.git
cd DashboardAnalytics/DashboardAnalyticsForJJ

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the `DashboardAnalyticsForJJ/` directory:

```env
ANTHROPIC_API_KEY="sk-ant-..."
```

Optional overrides:

```env
SPEND_FILE_PATH="/path/to/Enterprise_Spend_Jan_2026.xlsx"
RISK_FILE_PATH="/path/to/RiskRegisterSample.xlsx"
EXPORT_DIR="/path/to/Reports"
ANALYTICS_REFERENCE_DATE="2026-04-08"
```

### Run

```bash
reflex run
```

The dashboard opens at **http://localhost:3000**.

### Run Tests

```bash
pytest tests/ -v
```

---

## Data Inputs

| File | Purpose |
|---|---|
| `Enterprise_Spend_Jan_2026.xlsx` | Purchase order headers and spend detail with variance data |
| `RiskRegisterSample.xlsx` | Risk register with ownership, status, severity, and aging |

Place both files in the project root (or configure paths via `.env`).

---

## Tech Stack

| Component | Technology |
|---|---|
| UI Framework | [Reflex](https://reflex.dev/) 0.8.28+ |
| Charts | [Plotly](https://plotly.com/python/) 6.6 |
| AI Orchestration | [LangGraph](https://langchain-ai.github.io/langgraph/) 1.1.6 |
| LLM | [Anthropic Claude 3.5 Sonnet](https://www.anthropic.com/) |
| Data Processing | [Pandas](https://pandas.pydata.org/) 3.0 + [OpenPyXL](https://openpyxl.readthedocs.io/) |
| PDF Export | [ReportLab](https://www.reportlab.com/) 4.4 |
| Testing | [pytest](https://pytest.org/) 9.0 |

---

## License

This project is proprietary. All rights reserved.
