# Project Memory Notes

- Project identity: DashboardAnalyticsForJJ — Claude-powered Spend, Variance, and Risk Analytics Dashboard.
- Primary data sources: Enterprise_Spend_Jan_2026.xlsx and RiskRegisterSample.xlsx.
- Canonical spend dimensions: PO_Number, Sector, PO_Status, Addressable_Flag, Root_Cause, Last_Updated_Timestamp.
- Canonical risk dimensions: Risk_ID, Risk Owner, Risk Status, Risk Category, Risk Level, Open Date, Closed Date, Days_Open.
- Core dashboard workflow: load_data to compute_metrics to generate_charts to update_state.
- Agentic actions: explain variance, email risk owner, export filtered report, upload files, scrape metadata, AI column analysis.
- Sidebar hosts Claude chat. Main area hosts dashboard metrics and visualizations.
- Multi-file upload supports Excel (.xlsx/.xls), JSON, and PDF with drag-and-drop.
- File scraper extracts column names, data types, sample values, sheet structure (Excel), keys (JSON), headings (PDF).
- Claude column analyzer classifies columns as spend, variance, risk, time, identifier, metadata, or irrelevant.
- Column analysis includes confidence levels, schema mapping, and actionable recommendations.
- Deterministic fallback for column analysis uses keyword matching when Claude API is unavailable.
- Left navigation is required to support collapsible behavior with persistent open or closed preference.
- Chat panel must support collapsible, dock-right, and floating draggable behavior with persisted state.
- Chart surfaces must use consistent minimal cards, responsive grid layout, and expandable large-view interactions.
- Dashboard analytics must support Today, This Week, and This Month range modes with selector-driven recomputation.
- Dashboard must include structured title bar and compact collapsible filter UX with active chips.
- Comprehensive UI remediation active: full-label icon nav, compact expandable filters, dockable chat UX, meaningful KPI cards, interactive styled charts, consistent spacing rhythm, time-range-aware analytics, functional chart modal expansion, and styled predictive insight card placement.
