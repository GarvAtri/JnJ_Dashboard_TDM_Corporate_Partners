import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import anthropic
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from datetime import datetime
import json
from pathlib import Path

st.markdown("""
<style>

/* Sidebar inputs, selects, textareas */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    background-color: #2b3140 !important;
    color: #e5e7eb !important;
    border: 1px solid #4b5563 !important;
}

/* Placeholder text */
section[data-testid="stSidebar"] input::placeholder {
    color: #9ca3af !important;
}

/* Dropdown (Streamlit uses BaseWeb, not native select) */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    background-color: #2b3140 !important;
    color: #e5e7eb !important;
}

/* Buttons (like Browse files) */
section[data-testid="stSidebar"] button {
    background-color: #2b3140 !important;
    color: #e5e7eb !important;
    border: 1px solid #4b5563 !important;
}

</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="TDM · Signal", layout="wide", initial_sidebar_state="expanded")

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600,700,800&display=swap');

:root{
  --accent:#ff3355;
  --accent-dim:rgba(255,51,85,0.35);
  --cyan:#5eead4;
  --cyan-dim:rgba(94,234,212,0.12);
  --muted:#4b5563; /* darker muted */
  --text:#e8eef7;
  --bg0:#03050c;
  --bg1:#060a14;
  --card:rgba(8,12,24,0.72);
  --border:rgba(255,255,255,0.08);
  --glow:0 0 60px rgba(255,51,85,0.08);
  font-family:'Outfit',system-ui,sans-serif;
}

html, body, .stApp, [data-testid="stAppViewContainer"]{
  font-family:'Outfit',system-ui,sans-serif;
  color:var(--text);
}

.card{background:linear-gradient(155deg, rgba(255,255,255,0.04) 0%, rgba(0,0,0,0.25) 100%);border-radius:16px;padding:22px;border:1px solid var(--border);backdrop-filter:blur(20px)}
.section-title{font-family:'JetBrains Mono',monospace;font-size:11px;text-transform:uppercase;color:var(--muted);margin-bottom:12px}
.small{font-size:13px;color:var(--muted)}

.kpi-card{background:rgba(255,255,255,0.02);padding:12px;border-radius:10px}
.kpi-value{font-family:'JetBrains Mono',monospace;font-size:22px;font-weight:700;color:#f8fafc}
.kpi-label{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--muted);margin-top:6px}

.bubble{padding:10px 14px;border-radius:12px;color:#111827}
.muted{color:var(--muted);font-size:12px}

/* Left rail styling */
.left-rail{padding-top:6px}
.left-rail .stRadio>div{background:transparent}
.left-rail .stRadio label{display:block;padding:8px 10px;border-radius:8px;color:#cbd5e1}
.left-rail .stRadio label[aria-checked="true"]{background:#374151;color:#ffffff}

@media (max-width:720px){.main .block-container{padding-top:0.5rem}.hero-panel{padding:18px}}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# Chat history file path (persist across sessions)
ROOT = Path(__file__).parent
CHAT_HISTORY_FILE = ROOT / "chat_history.json"


def load_chat_history():
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
    except Exception:
        return []
    return []


def save_chat_history(msgs):
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as fh:
            json.dump(msgs, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ---- Top hero / tabs ----
        with st.spinner("Thinking..."):
            try:
                # Prefer concise, final answers from the LLM: add a small system instruction
                instruction = (
                    "You are a helpful data assistant. Provide a concise, final answer to the user's question. "
                    "If the user explicitly asks for steps or methods, include them after the concise answer. "
                    "Avoid returning only instructions for how to compute things; return the actual result or summary when possible."
                )
                if llm_provider == 'Gemini' and st.session_state.get('model_name'):
                    model = genai.GenerativeModel(st.session_state.model_name)
                    ctx = ''
                    if df is not None:
                        ctx = f"\nContext: columns = {list(df.columns)}"
                    prompt = f"{instruction}\nUser question: {prompt_text}{ctx}"
                    resp = model.generate_content(prompt)
                    text = getattr(resp, 'text', None) or str(resp)
                elif llm_provider == 'Claude' and st.session_state.get('claude_api_key'):
                    client = anthropic.Client(api_key=st.session_state.claude_api_key)
                    claude_prompt = f"System: {instruction}\nHuman: {prompt_text}\nAssistant:"
                    resp = client.completions.create(model=st.session_state.claude_model, prompt=claude_prompt, max_tokens_to_sample=300)
                    text = resp.completion
                else:
                    text = "No valid LLM configured."

                ts2 = datetime.utcnow().isoformat()
                st.session_state.messages.append({'role': 'assistant', 'content': text, 'ts': ts2})
                save_chat_history(st.session_state.messages)
            except Exception as e:
                ts2 = datetime.utcnow().isoformat()
                st.session_state.messages.append({'role': 'assistant', 'content': f'Error: {e}', 'ts': ts2})
                save_chat_history(st.session_state.messages)

    # Rerun so UI updates and shows the new messages immediately (some Streamlit
    # builds don't expose experimental_rerun; fall back gracefully)
    try:
        st.experimental_rerun()
    except Exception:
        # If rerun is not available, do nothing — Streamlit will rerun after
        # the current script execution completes on user interaction.
        pass


def generate_personalized_suggestions(messages, df, max_suggestions=6):
    """Create simple personalized suggestions based on the last exchange and dataset.

    This is a heuristic generator (no external LLM calls) so it's fast and
    deterministic. It looks at the last assistant message and dataset columns to
    propose relevant next steps.
    """
    suggestions = []
    last_user = None
    last_assistant = None
    for m in reversed(messages or []):
        if not last_assistant and m.get('role') == 'assistant':
            last_assistant = m.get('content','')
        if not last_user and m.get('role') == 'user':
            last_user = m.get('content','')
        if last_user and last_assistant:
            break

    # Basic generic suggestions
    suggestions.append('Summarize the dataset')
    suggestions.append('Find missing values')
    if last_assistant:
        suggestions.append('Explain the previous response')

    # dataset-driven suggestions
    if df is not None:
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
        if num_cols:
            for c in num_cols[:3]:
                suggestions.append(f'Show distribution of {c}')
        if len(num_cols) >= 2:
            suggestions.append(f'Compare {num_cols[0]} vs {num_cols[1]}')
        if cat_cols:
            for c in cat_cols[:2]:
                suggestions.append(f'Count categories of {c}')

    # deduplicate while preserving order
    seen = set()
    out = []
    for s in suggestions:
        if s not in seen:
            out.append(s)
            seen.add(s)
        if len(out) >= max_suggestions:
            break
    return out


def perform_local_action(prompt_text, df):
    """Perform a small set of local, deterministic actions instead of calling the LLM.

    Returns a string (possibly containing HTML) to present as the assistant reply,
    or None if no local action matched.
    """
    if df is None:
        return "No dataset loaded. Upload a CSV or choose a sample to run this action."

    p = prompt_text.lower().strip()

    # Missing values report
    if 'missing' in p or 'missing values' in p or p.startswith('find missing') or p.startswith('show missing'):
        mv_count = df.isnull().sum()
        mv_pct = (mv_count / len(df) * 100).round(2)
        summary = pd.DataFrame({'Missing Count': mv_count, 'Missing %': mv_pct}).reset_index()
        summary.columns = ['column', 'missing_count', 'missing_pct']
        # keep only columns with missing values
        summary = summary[summary['missing_count'] > 0].sort_values('missing_count', ascending=False)

        if summary.empty:
            return "No missing values detected in the current dataset. ✅"

        # preview rows that have any missing values (limit 50)
        preview = df[df.isnull().any(axis=1)].head(50).copy()
        # convert to HTML for inline display inside assistant bubble
        try:
            preview_html = preview.to_html(index=False, classes='dataframe', justify='left')
        except Exception:
            preview_html = ''

        # build a simple HTML response: counts + table preview
        rows_html = summary.to_html(index=False, classes='dataframe', justify='left')
        text = f"<div><strong>Missing values summary</strong></div>\n{rows_html}\n"
        if not preview_html:
            text += "<div style='margin-top:8px' class='small'>No preview available.</div>"
        else:
            text += f"<div style='margin-top:8px'><strong>Rows with missing values (preview)</strong>{preview_html}</div>"
        return text

    # no local action matched
    return None


def handle_prompt_and_respond(prompt_text):
    """Helper to record a user prompt and either run a local action or call the LLM.

    This mirrors the behavior of the chat input handler so suggestion buttons can
    invoke the same flow.
    """
    ts = datetime.utcnow().isoformat()
    st.session_state.messages.append({'role': 'user', 'content': prompt_text, 'ts': ts})
    save_chat_history(st.session_state.messages)

    # Fast local action first
    local = perform_local_action(prompt_text, st.session_state.get('df', None))
    if local is not None:
        ts2 = datetime.utcnow().isoformat()
        st.session_state.messages.append({'role': 'assistant', 'content': local, 'ts': ts2})
        save_chat_history(st.session_state.messages)
        try:
            st.experimental_rerun()
        except Exception:
            pass
        return

    # Otherwise call the configured LLM
    with st.spinner("Thinking..."):
        try:
            if st.session_state.get('llm_provider', llm_provider) == 'Gemini' and st.session_state.get('model_name'):
                model = genai.GenerativeModel(st.session_state.model_name)
                ctx = ''
                df_local = st.session_state.get('df', None)
                if df_local is not None:
                    ctx = f"\nContext: columns = {list(df_local.columns)}"
                resp = model.generate_content(prompt_text + ctx)
                text = getattr(resp, 'text', None) or str(resp)
            elif st.session_state.get('llm_provider', llm_provider) == 'Claude' and st.session_state.get('claude_api_key'):
                client = anthropic.Client(api_key=st.session_state.claude_api_key)
                claude_prompt = f"Human: {prompt_text}\nAssistant:"
                resp = client.completions.create(model=st.session_state.claude_model, prompt=claude_prompt, max_tokens_to_sample=300)
                text = resp.completion
            else:
                text = "No valid LLM configured."

            ts2 = datetime.utcnow().isoformat()
            st.session_state.messages.append({'role': 'assistant', 'content': text, 'ts': ts2})
            save_chat_history(st.session_state.messages)
            try:
                st.experimental_rerun()
            except Exception:
                pass
        except Exception as e:
            st.error(f"LLM error: {e}")

# ---- Sidebar: controls ----
with st.sidebar:
    st.markdown(
        """<div style='text-align:left'>
        <div class='pill'>TDM · SIGNAL</div>
        <div style='margin-top:12px;font-size:13px;color:#8b98a8;line-height:1.45'>Connect a model, load a table, ship insight.</div>
    </div>""",
        unsafe_allow_html=True,
    )
    st.divider()

    # Upload and sample dataset shortcuts
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], help="Drag & drop a file or click to browse")
    st.markdown("---")
    st.markdown("<div class='small'>Or try a sample dataset:</div>", unsafe_allow_html=True)
    sample = st.selectbox("Load sample", ["—","Iris","Tips (seaborn)","Wine (sklearn)"])

    # Targeted styling: make the text inside the "Load sample" select element darker
    # This CSS only targets the select control with the aria-label "Load sample" inside the sidebar.
    # It changes only the text color and option text; it does NOT modify backgrounds or global styles.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [aria-label="Load sample"],
        [data-testid="stSidebar"] [aria-label="Load sample"] * {
            color: #374151 !important;
        }
        /* Ensure option items in the dropdown also use the darker color */
        [data-testid="stSidebar"] [aria-label="Load sample"] option {
            color: #374151 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    # LLM Provider selection
    llm_provider = st.selectbox("LLM Provider", ["Gemini", "Claude"], index=0)
    st.session_state.llm_provider = llm_provider

    if llm_provider == "Gemini":
        # Gemini API & model controls
        if "gemini_api_key" not in st.session_state:
            st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
        api_key = st.text_input("Gemini API Key", value=st.session_state.gemini_api_key, type="password")
        if api_key:
            st.session_state.gemini_api_key = api_key
            try:
                genai.configure(api_key=api_key)
            except Exception:
                pass

        if "model_name" not in st.session_state:
            st.session_state.model_name = ""
        mn = st.text_input("Model name", value=st.session_state.model_name, placeholder="Leave blank and click 'List models' to fetch")
        if mn:
            st.session_state.model_name = mn

        if st.button("List models"):
            if not st.session_state.gemini_api_key:
                st.error("Set your Gemini API key first")
            else:
                try:
                    models = [ (getattr(m,'name',None) or str(m)) for m in genai.list_models() ]
                    if models:
                        chosen = st.selectbox("Available models", models, key="_models_list")
                        st.session_state.model_name = chosen
                        st.success(f"Selected {chosen}")
                except Exception as e:
                    st.error(f"ListModels failed: {e}")

    elif llm_provider == "Claude":
        # Claude API & model controls
        if "claude_api_key" not in st.session_state:
            st.session_state.claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        api_key = st.text_input("Claude API Key", value=st.session_state.claude_api_key, type="password")
        if api_key:
            st.session_state.claude_api_key = api_key

        if "claude_model" not in st.session_state:
            st.session_state.claude_model = "claude-3-sonnet-20240229"
        claude_models = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]
        selected_model = st.selectbox("Claude Model", claude_models, index=claude_models.index(st.session_state.claude_model) if st.session_state.claude_model in claude_models else 0)
        st.session_state.claude_model = selected_model

    st.markdown("---")
    st.markdown("<div class='small'>Visualization options</div>", unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Light","Dark"], index=0)
    # LLM response tuning
    st.markdown("---")
    st.markdown("<div class='small'>AI response settings</div>", unsafe_allow_html=True)
    response_style = st.selectbox("Response style", ["Concise","Detailed","Step-by-step"], index=0)
    st.session_state.response_style = response_style
    temp = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.05)
    st.session_state.temperature = float(temp)
    st.session_state.llm_interpret_local = st.checkbox("Use LLM to interpret local analyses", value=False)
    st.markdown("---")
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            if k not in ["gemini_api_key","model_name"]:
                del st.session_state[k]
        try:
            st.experimental_rerun()
        except Exception:
            pass

    # Chat history management in sidebar
    st.markdown("---")
    if st.button("Export chat history"):
        msgs = load_chat_history()
        if msgs:
            st.download_button("Download JSON", json.dumps(msgs, ensure_ascii=False, indent=2), file_name="chat_history.json", mime="application/json")
        else:
            st.info("No chat history to export.")

    if st.button("Clear chat history"):
        # clear persisted file and in-memory state
        try:
            if CHAT_HISTORY_FILE.exists():
                CHAT_HISTORY_FILE.unlink()
        except Exception:
            pass
        if 'messages' in st.session_state:
            st.session_state.messages = []
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ---- Load data ----
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
else:
    # load sample
    if sample == "Iris":
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        st.session_state.df = df
    elif sample == "Tips (seaborn)":
        import seaborn as sns
        df = sns.load_dataset('tips')
        st.session_state.df = df
    elif sample == "Wine (sklearn)":
        from sklearn.datasets import load_wine
        wine = load_wine(as_frame=True)
        df = wine.frame
        st.session_state.df = df

if 'df' in st.session_state and st.session_state.df is not None:
    df = st.session_state.df.copy()

# High-level dataset metrics for hero + plots
num_rows = int(df.shape[0]) if df is not None else 0
num_cols = int(df.shape[1]) if df is not None else 0
missing_cells = int(df.isna().sum().sum()) if df is not None else 0
total_cells = num_rows * num_cols if df is not None else 0
missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells else 0.0
numeric_cols_list = df.select_dtypes('number').columns.tolist() if df is not None else []
numeric_count = len(numeric_cols_list)

# Simple data quality score for display (0–100)
if df is not None and total_cells:
    density_score = max(0.0, 100.0 - missing_pct)  # less missing → higher
    width_score = min(100.0, 20.0 + num_cols * 2.0)  # more columns → slightly higher
    quality_score = int(round((density_score * 0.7) + (width_score * 0.3)))
else:
    quality_score = 0

# Plotly theme based on sidebar selection
plotly_template = 'plotly_white' if ('theme' not in globals() and locals().get('theme','Light') == 'Light') or (globals().get('theme','Light') == 'Light') or (locals().get('theme','Light') == 'Light' and theme == 'Light') else 'plotly_dark'

# Load persisted chat history into session state if available
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat_history()

# ---- Top hero / tabs ----
hero_left, hero_right = st.columns([4, 1.4])
with hero_left:
    chip_html = ""
    if df is not None:
        chip_html = f"""
            <div class="chip-row">
              <div class="chip"><strong>Rows</strong> {num_rows:,}</div>
              <div class="chip"><strong>Cols</strong> {num_cols:,}</div>
              <div class="chip"><strong>Numeric</strong> {numeric_count}</div>
              <div class="chip"><strong>Missing</strong> {missing_pct:.2f}%</div>
              <div class="chip"><strong>Quality</strong> {quality_score}/100</div>
            </div>"""
    st.markdown(
        f"""
        <div class="hero-panel">
          <div class="inner">
            <div class="pill">Signal · LLM console</div>
            <div class="hero-title">TDM Data Studio</div>
            <div class="hero-sub">Upload or sample a dataset, explore structure and charts, then steer analysis with natural language — all in one glass panel.</div>
            {chip_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    if df is not None:
        csv = df.head(1000).to_csv(index=False)
        st.download_button("Export CSV sample", csv, "sample.csv", "text/csv", use_container_width=True)

# ----- Analyze section -----
st.markdown("<div class='surface-panel'>", unsafe_allow_html=True)
st.markdown("<h2 class='fx-h2'>Analyze</h2>", unsafe_allow_html=True)
if df is None:
    st.info("Upload data or choose a sample to analyze with the LLM.")
else:
    st.markdown("<div class='section-title'>Ask the model</div>", unsafe_allow_html=True)
    question = st.text_area("Ask the LLM about this data", placeholder="Ask about trends, anomalies, summaries, feature importance...", label_visibility="collapsed")
    if st.button("Run analysis", type="primary"):
        if not st.session_state.gemini_api_key:
            st.error("Set Gemini API key in sidebar first.")
        elif not st.session_state.model_name:
            st.error("Select a model name in the sidebar (List models to fetch).")
        else:
            with st.spinner("Contacting LLM..."):
                try:
                    instruction = (
                        "You are a helpful data assistant. Provide a concise, final answer to the user's question. "
                        "If the user explicitly asks for steps or methods, include them after the concise answer. "
                        "Avoid returning only instructions for how to compute things; return the actual result or summary when possible."
                    )
                    prompt_text = question
                    if llm_provider == 'Gemini' and st.session_state.get('model_name'):
                        model = genai.GenerativeModel(st.session_state.model_name)
                        ctx = ''
                        if df is not None:
                            ctx = f"\nContext: columns = {list(df.columns)}"
                        prompt = f"{instruction}\nUser question: {prompt_text}{ctx}"
                        resp = model.generate_content(prompt)
                        text = getattr(resp, 'text', None) or str(resp)
                    elif llm_provider == 'Claude' and st.session_state.get('claude_api_key'):
                        client = anthropic.Client(api_key=st.session_state.claude_api_key)
                        claude_prompt = f"System: {instruction}\nHuman: {prompt_text}\nAssistant:"
                        resp = client.completions.create(model=st.session_state.claude_model, prompt=claude_prompt, max_tokens_to_sample=300)
                        text = resp.completion
                    else:
                        text = "No valid LLM configured."

                    ts2 = datetime.utcnow().isoformat()
                    st.session_state.messages.append({'role': 'assistant', 'content': text, 'ts': ts2})
                    save_chat_history(st.session_state.messages)
                except Exception as e:
                    ts2 = datetime.utcnow().isoformat()
                    st.session_state.messages.append({'role': 'assistant', 'content': f'Error: {e}', 'ts': ts2})
                    save_chat_history(st.session_state.messages)
            # close Analyze panel
            st.markdown("</div>", unsafe_allow_html=True)

            # If toggled, show personalized suggestions generated from recent messages + dataset
            if st.session_state.show_suggestions:
                suggs = generate_personalized_suggestions(st.session_state.get('messages', []), df, max_suggestions=8)
                if suggs:
                    btn_cols = st.columns(len(suggs))
                    for i, s in enumerate(suggs):
                        if btn_cols[i].button(s, key=f'sugg_popup_{i}'):
                            handle_prompt_and_respond(s)

            # Chat input
            if prompt := st.chat_input("Start conversation..."):
                # record user message with timestamp and persist
                ts = datetime.utcnow().isoformat()
                st.session_state.messages.append({'role':'user','content':prompt,'ts':ts})
                save_chat_history(st.session_state.messages)
                st.chat_message('user').markdown(prompt)

                if llm_provider == 'Gemini' and not st.session_state.get('model_name'):
                    st.error("Select a Gemini model in the sidebar first.")
                else:
                    # Call appropriate LLM
                    with st.spinner("Thinking..."):
                        try:
                            if llm_provider == 'Gemini' and st.session_state.get('model_name'):
                                model = genai.GenerativeModel(st.session_state.model_name)
                                ctx = ''
                                if df is not None:
                                    ctx = f"\nContext: columns = {list(df.columns)}"
                                resp = model.generate_content(prompt + ctx)
                                text = getattr(resp,'text',None) or str(resp)
                            elif llm_provider == 'Claude' and st.session_state.get('claude_api_key'):
                                client = anthropic.Client(api_key=st.session_state.claude_api_key)
                                claude_prompt = f"Human: {prompt}\nAssistant:"
                                resp = client.completions.create(model=st.session_state.claude_model, prompt=claude_prompt, max_tokens_to_sample=300)
                                text = resp.completion
                            else:
                                text = "No valid LLM configured."

                            # save assistant response with timestamp
                            ts2 = datetime.utcnow().isoformat()
                            st.session_state.messages.append({'role':'assistant','content':text,'ts':ts2})
                            save_chat_history(st.session_state.messages)

                            with st.chat_message('assistant'):
                                st.markdown(text)
                        except Exception as e:
                            st.error(f"Chat error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

