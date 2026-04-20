
# DashboardAnalyticsForJohnson-Johnson

Authors: Pragathi and Manasi

This repository contains a Streamlit-based dashboard application (main file: `claude_dashboard 1.py`) for exploring uploaded tabular data, visualizations, and a small assistant/chat interface. The app has an aesthetic dark theme by default and a Light/Dark toggle in the left sidebar. It also supports uploading up to 3 files at once and runs compact per-file analyses.

This README collects step-by-step setup and run instructions, common troubleshooting steps, and notes about the UI and configuration.

## Requirements

- macOS (these steps are written for macOS zsh shell)
- Python 3.9+ (the project was built with Python 3.9; note that some packages warn about Python 3.9 EOL — upgrading to 3.10+ is recommended)
- A virtual environment is recommended (venv/virtualenv/conda).

Recommended packages used by the app include: streamlit, pandas, numpy, plotly, scikit-learn, seaborn, and optionally LLM client libraries (Anthropic, Google AI SDK). A `requirements.txt` may be present in the repo; if not, install the essentials listed below.

## Quick setup (recommended)

1. Open a Terminal (zsh) and change to the project folder:

```zsh
cd /Users/pragathi/tdm
```

2. Create and activate a virtual environment (venv):

```zsh
# create venv
python3 -m venv .venv

# activate venv
source .venv/bin/activate
```

3. Install dependencies. If the repo includes `requirements.txt`, use it. Otherwise install the minimal set manually:

```zsh
# If requirements.txt exists
pip install -r requirements.txt

# Otherwise install essentials
pip install streamlit pandas numpy plotly scikit-learn seaborn
# If you use LLM integrations, install them as needed (Anthropic/Google SDKs). Be aware of deprecation notices.
```

4. (Optional) Pin versions or create a lockfile if you plan to reproduce the environment across machines.

## Running the app

Start the Streamlit app from the project root. The main file is `claude_dashboard 1.py`.

```zsh
# run on default port (8501) or specify a port (8503 used during development sometimes)
streamlit run "claude_dashboard 1.py" --server.port 8503
```

When Streamlit starts, it will print a Local URL such as `http://localhost:8503`. Open that in your browser.

Notes:
- If the server prints warnings about Python or libraries (e.g., Python 3.9 deprecation warnings), they are informative and not fatal.
- If you change the code, Streamlit will auto-reload the app.

## UI overview

- Left sidebar:
  - Theme selector (Light / Dark). The app injects scoped CSS when you change this selector to switch the UI colors.
  - Multi-file uploader (up to 3 files). Uploaded files are parsed into pandas DataFrames and stored in `st.session_state['uploaded_files']`.
  - `Active file` selector to choose which uploaded dataset to analyze.
  - Controls for LLM provider, model selection, temperature, and API key fields (if LLM integrations are enabled).

- Main area:
  - Overview, Visualize, Chat tabs. Per-file compact analyses are shown by `analyze_dataframe()` helper.

## Theme behavior and CSS notes

- The app uses a base dark theme injected via a top-of-file `_CSS` string. Additional scoped `st.markdown("<style>...</style>")` blocks are used to fix specific widget appearances in the sidebar and textarea controls.
- The Theme toggle in the sidebar injects light-mode CSS overrides when set to Light. If you do not see the expected light colors, try a full browser refresh or restart the Streamlit server to ensure CSS injections applied.

Programmatic verification (developer):

```zsh
# Example: fetch HTML served by Streamlit to check for specific injected CSS tokens
curl -s http://localhost:8503 | grep -E "#2b3140|--text: #0f172a" || true
```

If you see `#2b3140` the dark-themed CSS is present; if you see `--text: #0f172a` the light-mode root variable overrides are present.

## Common issues and troubleshooting

- StreamlitAPIException: "st.session_state.<key> cannot be modified after the widget with key <key> is instantiated." —
  - Cause: code assigns to a `st.session_state` key immediately after creating a widget with the same key. Fix: remove assignment or set defaults before creating the widget.

- IndentationError / SyntaxError when editing `claude_dashboard 1.py` —
  - The file has grown with many edits; ensure you preserve Python indentation. Use `python -m py_compile "claude_dashboard 1.py"` to check syntax.

- NotOpenSSLWarning (urllib3) or OpenSSL mismatch —
  - Not critical for local development. If you see SSL-related errors during runtime network calls, ensure your system OpenSSL and Python build are compatible.

- Deprecation warnings for Google/LLM libraries —
  - Some LLM integration libraries may be deprecated. If you rely on them, consult the provider docs and replace with supported SDKs.

## Debugging steps (developer)

1. Validate syntax quickly:

```zsh
python -m py_compile "claude_dashboard 1.py"
```

2. Run the app with verbose logs if you need more output (Streamlit prints logs to the terminal):

```zsh
streamlit run "claude_dashboard 1.py" --server.port 8503
```

3. When making CSS changes, the browser cache can hide results. Use hard reload (Cmd+Shift+R) or open a private window.

4. If you're programmatically validating the served HTML, allow the server to fully start before fetching HTML. Add a short sleep or retry loop in scripts.

## Files of interest

- `claude_dashboard 1.py` — main Streamlit application
- `.venv/` — recommended virtual environment location (not committed)
- `chat_history.json` (if present at runtime) — persisted chat history used by the app
- `requirements.txt` — pip dependencies (if present)

## Optional developer tips

- Persisting theme choice: if you'd like theme choice persisted between runs, add a small persistence (e.g., write the selected theme to a file in the project root or to `st.experimental_set_query_params`) and load it on startup.
- Tests: add a small pytest test to validate `analyze_dataframe()` behavior and run quickly before manual QA.

## Example commands

```zsh
# Activate venv
source .venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run app
streamlit run "claude_dashboard 1.py" --server.port 8503

# Check python syntax
python -m py_compile "claude_dashboard 1.py"

# Search for injected CSS in served HTML (after server started)
curl -s http://localhost:8503 | grep -E "#2b3140|--text: #0f172a" || true
```

## Contact / Next steps

If you'd like I can:

- Add a small `requirements.txt` pinned from your virtual environment.
- Add a small smoke test (pytest) that imports the app helpers and runs basic checks.
- Persist theme selection across runs.

If you'd like any of these, tell me which and I'll implement them next.

---
Generated on: April 19, 2026
# LLM Powered Data Dashboard

This is a Streamlit app (`dashboard_app.py`) that lets you upload a CSV/Excel file, explore it, ask an LLM (Gemini) for analysis, and chat with the assistant using the Gemini API.

Quick start

1. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Set your Gemini API key in the environment (or enter it in the sidebar when the app runs):

```bash
export GEMINI_API_KEY="your_api_key_here"
```

4. Run the app:

```bash
streamlit run dashboard_app.py
```

Notes
- The app uses the Google Generative AI SDK (imported as `google.generativeai` in the code). If the SDK API changes, you may need to adjust the calls.
- For Excel file upload support we include `openpyxl` in `requirements.txt`.