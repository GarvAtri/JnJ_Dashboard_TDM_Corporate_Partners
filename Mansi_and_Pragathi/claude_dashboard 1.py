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

# Inserted sidebar CSS to darken white controls in the left sidebar
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



# Scoped fixes requested by user: export button text and "Ask the model" textarea
st.markdown("""
<style>
/* Export CSV sample button text */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stDownloadButton button,
section[data-testid="stSidebar"] .stButton button {
    color: #e5e7eb !important;
}

/* Main page download/export button text if it is outside sidebar */
.stDownloadButton button,
.stButton button {
    color: #e5e7eb !important;
}

/* Ask the model text area */
textarea,
.stTextArea textarea {
    background-color: #2b3140 !important;
    color: #e5e7eb !important;
    border: 1px solid #ef4444 !important;
}

/* Ask the model placeholder */
textarea::placeholder,
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
    opacity: 1 !important;
}

/* If Streamlit wraps the text area in extra containers, make inner text readable too */
.stTextArea * {
    color: inherit !important;
}

/* If the Export CSV sample button inherits opacity or is disabled, ensure label is visible */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] .stDownloadButton button,
section[data-testid="stSidebar"] .stButton button {
    opacity: 1 !important;
    color: #e5e7eb !important;
}

/* Export CSV sample button text fix */
.stDownloadButton button,
button[kind="secondary"],
button[kind="primary"] {
    color: #1f2937 !important;
    opacity: 1 !important;
}

</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(page_title="TDM · Signal", layout="wide", initial_sidebar_state="expanded")

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700;800&display=swap');

/* Signal console — deep space + neon edge */
:root{
  --accent:#ff3355;
  --accent-dim:rgba(255,51,85,0.35);
  --cyan:#5eead4;
  --cyan-dim:rgba(94,234,212,0.12);
  --muted:#8b98a8;
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

.stApp{
  background:
    radial-gradient(ellipse 100% 80% at 50% -30%, rgba(255,51,85,0.14), transparent 50%),
    radial-gradient(ellipse 80% 50% at 100% 20%, rgba(94,234,212,0.08), transparent 45%),
    radial-gradient(ellipse 70% 40% at 0% 60%, rgba(99,102,241,0.06), transparent 50%),
    linear-gradient(165deg, var(--bg1) 0%, var(--bg0) 55%, #020308 100%) !important;
  background-attachment:fixed;
}

/* Grid sits behind all app content (sidebar was painting under this in some builds) */
.stApp::before{
  content:"";
  pointer-events:none;
  position:fixed;
  inset:0;
  background-image:
    linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(ellipse 90% 70% at 50% 30%, black 20%, transparent 75%);
  opacity:0.35;
  z-index:0;
}

[data-testid="stHeader"]{
  background:rgba(3,5,12,0.75)!important;
  backdrop-filter:blur(12px);
  border-bottom:1px solid var(--border);
}

[data-testid="stSidebar"]{
  position:relative;
  z-index:4!important;
  background:linear-gradient(175deg, rgba(8,10,22,0.98) 0%, rgba(3,5,14,0.99) 100%)!important;
  border-right:1px solid rgba(255,255,255,0.1);
  box-shadow:8px 0 40px rgba(0,0,0,0.45);
  color:#e8eef7!important;
}

[data-testid="stSidebar"] .block-container{padding-top:1.25rem}

/* Sidebar: force readable labels (Streamlit widgets often use their own colors) */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="checkbox"] label{
  color:#e8eef7!important;
}

[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stSlider label p{
  color:#e8eef7!important;
}

.main{
  position:relative;
  z-index:2;
}

.main .block-container{
  padding-top:1.1rem;
  padding-bottom:2.5rem;
  max-width:1180px;
  position:relative;
  z-index:1;
}

/* Alerts & info — glass, not default blue */
div[data-testid="stAlert"]{
  background:linear-gradient(135deg, rgba(94,234,212,0.06), rgba(8,12,24,0.85))!important;
  border:1px solid rgba(94,234,212,0.2)!important;
  border-radius:12px!important;
  color:var(--text)!important;
}
div[data-testid="stAlert"] p, div[data-testid="stAlert"] div{
  color:var(--text)!important;
}

/* Expanders */
.streamlit-expanderHeader{
  font-weight:600!important;
  color:var(--text)!important;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]{
  background:var(--accent)!important;
  box-shadow:0 0 12px var(--accent-dim);
}

/* File drop zone */
[data-testid="stFileUploader"] section{
  border-radius:12px!important;
  border:1px dashed rgba(255,255,255,0.12)!important;
  background:rgba(5,8,18,0.6)!important;
}

/* Dataframes — darker chrome */
[data-testid="stDataFrame"]{
  border-radius:10px;
  overflow:hidden;
  border:1px solid var(--border);
}

/* Glass cards */
.card{
  background:linear-gradient(155deg, rgba(255,255,255,0.04) 0%, rgba(0,0,0,0.25) 100%);
  border-radius:16px;
  padding:22px 22px 20px;
  border:1px solid var(--border);
  box-shadow:
    var(--glow),
    0 24px 48px rgba(0,0,0,0.5),
    inset 0 1px 0 rgba(255,255,255,0.04);
  backdrop-filter:blur(20px);
}

.surface-panel{
  margin-bottom:1.5rem;
  padding:1.25rem 1.35rem 1.35rem;
  border-radius:16px;
  border:1px solid rgba(255,255,255,0.07);
  background:linear-gradient(160deg, rgba(255,255,255,0.03), rgba(5,8,20,0.85));
  box-shadow:0 20px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04);
}

.fx-h2{
  font-family:'Outfit',sans-serif;
  font-size:1.05rem;
  font-weight:700;
  letter-spacing:0.08em;
  text-transform:uppercase;
  margin:0 0 0.75rem;
  color:var(--text);
  border-left:3px solid var(--accent);
  padding-left:12px;
}

.section-title{
  font-family:'JetBrains Mono',ui-monospace,monospace;
  font-size:11px;
  text-transform:uppercase;
  letter-spacing:0.16em;
  color:var(--muted);
  margin-bottom:12px;
}

.small{
  font-size:13px;
  color:var(--muted);
  line-height:1.5;
}

/* Hero */
.hero-panel{
  position:relative;
  padding:22px 24px 20px;
  border-radius:18px;
  margin-bottom:1.35rem;
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.09);
  background:
    radial-gradient(ellipse 80% 120% at 100% 0%, rgba(255,51,85,0.18), transparent 55%),
    radial-gradient(ellipse 60% 80% at 0% 100%, rgba(94,234,212,0.1), transparent 50%),
    linear-gradient(145deg, rgba(12,16,32,0.95), rgba(4,6,14,0.98));
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.04) inset,
    0 28px 60px rgba(0,0,0,0.55),
    0 0 80px rgba(255,51,85,0.06);
}
.hero-panel::after{
  content:"";
  position:absolute;
  inset:0;
  background:repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.015) 2px, rgba(255,255,255,0.015) 3px);
  pointer-events:none;
  opacity:0.5;
}
.hero-panel .inner{position:relative;z-index:1}
.hero-title{
  font-family:'Outfit',sans-serif;
  font-size:clamp(1.35rem, 2.4vw, 1.75rem);
  font-weight:800;
  letter-spacing:-0.03em;
  margin:10px 0 6px;
  /* Solid text — gradient+transparent fill can disappear in some browsers */
  color:#f8fafc;
  text-shadow:0 0 28px rgba(94,234,212,0.2), 0 2px 12px rgba(0,0,0,0.45);
}
.hero-sub{font-size:14px;color:#b4c0d4;max-width:52ch;line-height:1.55}

.pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:5px 12px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.12);
  background:linear-gradient(120deg, rgba(255,51,85,0.25), rgba(15,23,42,0.9));
  font-family:'JetBrains Mono',monospace;
  font-size:10px;
  text-transform:uppercase;
  letter-spacing:0.2em;
  color:#f0f4ff;
}

.chip-row{
  display:flex;
  flex-wrap:wrap;
  gap:8px;
  margin-top:14px;
}
.chip{
  padding:7px 12px;
  border-radius:10px;
  background:rgba(0,0,0,0.35);
  border:1px solid rgba(255,255,255,0.1);
  font-family:'JetBrains Mono',monospace;
  font-size:11px;
  color:#cbd5e1;
}
.chip strong{
  color:#f1f5f9;
  margin-right:6px;
  font-weight:600;
}

/* Inputs & primary actions */
.stTextInput>div>div>input,
.stSelectbox>div>div>select,
textarea{
  border-radius:11px!important;
  padding:10px 12px!important;
  background:rgba(3,6,18,0.85)!important;
  border:1px solid rgba(148,163,184,0.28)!important;
  color:var(--text)!important;
  font-family:'Outfit',sans-serif!important;
}
.stTextInput>div>div>input:focus,
textarea:focus{
  border-color:rgba(94,234,212,0.45)!important;
  box-shadow:0 0 0 1px rgba(94,234,212,0.2)!important;
}

.stButton>button{
  border-radius:999px!important;
  padding:0.5rem 1.25rem!important;
  font-weight:600!important;
  font-family:'Outfit',sans-serif!important;
  border:1px solid rgba(255,255,255,0.14)!important;
  background:rgba(8,12,24,0.75)!important;
  color:var(--text)!important;
  box-shadow:none!important;
  transition:transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease!important;
}
.stButton>button:hover{
  border-color:rgba(94,234,212,0.35)!important;
  background:rgba(94,234,212,0.07)!important;
}
/* Primary (main CTAs) */
.stButton>button[kind="primary"]{
  border:1px solid rgba(255,51,85,0.55)!important;
  background:linear-gradient(135deg, rgba(255,51,85,0.98), rgba(180,30,60,0.92))!important;
  color:#fff!important;
  box-shadow:0 4px 24px rgba(255,51,85,0.28)!important;
}
.stButton>button[kind="primary"]:hover{
  border-color:#ff6b7a!important;
  box-shadow:0 6px 28px rgba(255,51,85,0.45)!important;
  transform:translateY(-1px);
}

/* Tabs — pill rail */
.stTabs [data-baseweb="tab-list"]{
  gap:6px;
  background:rgba(5,8,18,0.6)!important;
  padding:6px!important;
  border-radius:12px!important;
  border:1px solid rgba(255,255,255,0.06)!important;
}
.stTabs [data-baseweb="tab"]{
  border-radius:8px!important;
  font-weight:600!important;
  font-family:'Outfit',sans-serif!important;
  color:var(--muted)!important;
  padding:0.5rem 1rem!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(180deg, rgba(255,51,85,0.35), rgba(255,51,85,0.12))!important;
  color:#fff!important;
  box-shadow:0 0 20px rgba(255,51,85,0.2);
}

.main h1{
  font-family:'Outfit',sans-serif;
}

/* KPI — must stay in one flex parent (single HTML block in app) */
.kpi-grid{
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(130px, 1fr));
  gap:10px;
}
.kpi-card{
  padding:14px 14px 12px;
  border-radius:12px;
  background:
    radial-gradient(ellipse 80% 80% at 0% 0%, rgba(255,51,85,0.2), transparent 55%),
    linear-gradient(165deg, rgba(18,22,38,0.95), rgba(5,8,18,0.98));
  border:1px solid rgba(255,255,255,0.07);
  box-shadow:inset 0 1px 0 rgba(255,255,255,0.04);
  transition:transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover{
  transform:translateY(-2px);
  border-color:rgba(94,234,212,0.25);
}
.kpi-value{
  font-family:'JetBrains Mono',monospace;
  font-size:22px;
  font-weight:700;
  color:#f8fafc;
  font-variant-numeric:tabular-nums;
}
.kpi-label{
  font-family:'JetBrains Mono',monospace;
  font-size:10px;
  text-transform:uppercase;
  letter-spacing:0.14em;
  color:var(--muted);
  margin-top:6px;
}

.chat-window{
  background:linear-gradient(180deg, rgba(12,16,32,0.95), rgba(4,6,14,0.98));
  padding:14px;
  border-radius:14px;
  border:1px solid rgba(255,255,255,0.08);
}
.bubble{
  padding:10px 14px;
  border-radius:12px;
}
.muted{
  color:var(--muted);
  font-size:12px;
}

@media (max-width:720px){
  .main .block-container{padding-top:0.5rem}
  .hero-panel{padding:18px}
}
"""

st.html(f"<style>{_CSS}</style>")

# Chat history file path (persist across sessions)
ROOT = Path(__file__).parent
CHAT_HISTORY_FILE = ROOT / "chat_history.json"


# helper to load/save chat history
def load_chat_history():
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                return data if isinstance(data, list) else []
    except Exception:
        return []
    return []


def save_chat_history(messages):
    try:
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as fh:
            json.dump(messages, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass


# Helper to send a prompt and persist the assistant response
def handle_prompt_and_respond(prompt_text):
    # record user message
    ts = datetime.utcnow().isoformat()
    st.session_state.messages.append({'role': 'user', 'content': prompt_text, 'ts': ts})
    save_chat_history(st.session_state.messages)

    # try local action first (fast, deterministic)
    local_resp = perform_local_action(prompt_text, df)
    if local_resp is not None:
        # Optionally have the LLM interpret or summarize the local analysis
        if st.session_state.get('llm_interpret_local'):
            # build an instruction based on selected response style
            style = st.session_state.get('response_style', 'Concise')
            style_inst = 'Provide a concise interpretation.' if style == 'Concise' else ('Provide a detailed interpretation.' if style == 'Detailed' else 'Provide a step-by-step interpretation and actions.')
            interp_prompt = f"System: You are a data assistant. {style_inst} Interpret the following local analysis and provide clear action items:\n\n{local_resp}\n\nReturn the interpretation."
            try:
                # call LLM to interpret local result
                if llm_provider == 'Gemini' and st.session_state.get('model_name'):
                    model = genai.GenerativeModel(st.session_state.model_name)
                    resp = model.generate_content(interp_prompt)
                    text = getattr(resp, 'text', None) or str(resp)
                elif llm_provider == 'Claude' and st.session_state.get('claude_api_key'):
                    client = anthropic.Client(api_key=st.session_state.claude_api_key)
                    claude_prompt = f"System: {style_inst}\nHuman: Interpret the following local analysis:\n{local_resp}\nAssistant:"
                    resp = client.completions.create(model=st.session_state.claude_model, prompt=claude_prompt, max_tokens_to_sample=300)
                    text = resp.completion
                else:
                    text = local_resp
            except Exception:
                text = local_resp

            ts2 = datetime.utcnow().isoformat()
            st.session_state.messages.append({'role': 'assistant', 'content': text, 'ts': ts2})
            save_chat_history(st.session_state.messages)
        else:
            ts2 = datetime.utcnow().isoformat()
            st.session_state.messages.append({'role': 'assistant', 'content': local_resp, 'ts': ts2})
            save_chat_history(st.session_state.messages)
    else:
        # call appropriate LLM and append assistant reply
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

    # Theme selector (top of sidebar) — controls which CSS we inject for the UI.
    theme = st.selectbox("Theme", ["Dark", "Light"], index=0, key="ui_theme")

    # Inject dark-styled inputs only when Dark is selected (scoped to sidebar)
    if theme == "Dark":
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] select,
            [data-testid="stSidebar"] textarea,
            [data-testid="stSidebar"] button,
            [data-testid="stSidebar"] [role="combobox"],
            [data-testid="stSidebar"] [data-slot="select-trigger"],
            [data-testid="stSidebar"] [data-slot="select-value"] {
                background: #2b3140 !important;
                color: #e5e7eb !important;
                border: 1px solid #4b5563 !important;
            }

            [data-testid="stSidebar"] input::placeholder,
            [data-testid="stSidebar"] textarea::placeholder {
                color: #9ca3af !important;
                opacity: 1 !important;
            }

            [data-testid="stSidebar"] button span,
            [data-testid="stSidebar"] [role="combobox"] span,
            [data-testid="stSidebar"] [data-slot="select-trigger"] span,
            [data-testid="stSidebar"] [data-slot="select-value"] span {
                color: #e5e7eb !important;
            }

            [data-testid="stSidebar"] [data-placeholder="true"] { color: #9ca3af !important; }
            [data-testid="stSidebar"] .text-muted-foreground { color: #9ca3af !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # If user selects Light, inject a compact light-mode override for the whole app.
    if theme == "Light":
        st.markdown(
            """
            <style>
            :root{
              --text: #0f172a !important;
              --muted: #6b7280 !important;
              --bg0: #f8fafc !important;
              --bg1: #ffffff !important;
              --card: rgba(255,255,255,0.96) !important;
              --border: rgba(15,23,42,0.06) !important;
            }
            .stApp{
              background: linear-gradient(165deg, var(--bg1) 0%, var(--bg0) 55%, #ffffff 100%) !important;
              color: var(--text) !important;
            }
            [data-testid="stSidebar"]{
              background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)!important;
              color: var(--text) !important;
              border-right:1px solid var(--border) !important;
              box-shadow:none !important;
            }
            .card, .surface-panel, .hero-panel{
              background: var(--card) !important;
              color: var(--text) !important;
              border:1px solid var(--border) !important;
            }
            textarea, .stTextArea textarea {
              background-color: #ffffff !important;
              color: #0f172a !important;
              border: 1px solid #e5e7eb !important;
            }
            .muted, .kpi-label, .text-muted-foreground { color: var(--muted) !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Upload and sample dataset shortcuts
    uploaded_files = st.file_uploader("Upload CSV / Excel (up to 3 files)", type=["csv","xlsx","xls"], accept_multiple_files=True, help="Drag & drop files or click to browse (max 3)")
    st.markdown("---")
    st.markdown("<div class='small'>Or try a sample dataset:</div>", unsafe_allow_html=True)
    sample = st.selectbox("Load sample", ["—","Iris","Tips (seaborn)","Wine (sklearn)"])

    # If the user uploaded files, read up to 3 and store DataFrames in session state.
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Maximum 3 files allowed; only the first 3 will be processed.")
        files_to_use = uploaded_files[:3]
        uploaded_dfs = {}
        for uf in files_to_use:
            try:
                if uf.name.lower().endswith('.csv'):
                    df_i = pd.read_csv(uf)
                else:
                    df_i = pd.read_excel(uf)
                uploaded_dfs[uf.name] = df_i
            except Exception as e:
                st.error(f"Failed to read {uf.name}: {e}")

        if uploaded_dfs:
            # store the uploaded DataFrames and let the user pick an active file for analysis
            st.session_state.uploaded_files = uploaded_dfs
            active_name = st.selectbox("Active file", list(uploaded_dfs.keys()))
            st.session_state.active_file = active_name

    # Targeted styling: make the text inside the "Load sample" select element darker
    # This CSS only targets the select control with the aria-label "Load sample" inside the sidebar.
    # It changes only the text color and option text; it does NOT modify backgrounds or global styles.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [aria-label="Load sample"],
        [data-testid="stSidebar"] [aria-label="Load sample"] *,
        /* also target common select trigger/value slots used by custom select components */
        [data-testid="stSidebar"] [data-slot="select-trigger"],
        [data-testid="stSidebar"] [data-slot="select-value"],
        [data-testid="stSidebar"] [data-slot="select-trigger"] *,
        [data-testid="stSidebar"] [data-slot="select-value"] * {
            color: #1f2937 !important;
        }
        /* Ensure option items in the dropdown also use the darker color */
        [data-testid="stSidebar"] [aria-label="Load sample"] option {
            color: #1f2937 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    # LLM Provider selection
    # Scoped style: make selected values and placeholders inside the sidebar controls more readable
    # - selected value text: dark gray (#374151)
    # - placeholder text: medium-dark gray (#6b7280)
    # This CSS is strictly scoped to `[data-testid="stSidebar"]` so it doesn't affect global styles.
    st.markdown(
        """
        <style>
        /* Selected value text inside selectboxes in the sidebar */
        /* Selected value text inside selectboxes and custom select triggers in the sidebar */
        [data-testid="stSidebar"] .stSelectbox>div>div>select,
        [data-testid="stSidebar"] select,
        [data-testid="stSidebar"] [data-baseweb="select"] > div[role="combobox"],
        [data-testid="stSidebar"] [data-baseweb="select"] > div[role="button"],
        [data-testid="stSidebar"] [role="combobox"],
        [data-testid="stSidebar"] div[role="button"],
        [data-testid="stSidebar"] [data-slot="select-trigger"],
        [data-testid="stSidebar"] [data-slot="select-value"],
        [data-testid="stSidebar"] .SelectValue,
        [data-testid="stSidebar"] .SelectTrigger {
            color: #1f2937 !important;
        }

        [data-testid="stSidebar"] .stSelectbox>div>div>select option,
        [data-testid="stSidebar"] [data-baseweb="select"] option,
        [data-testid="stSidebar"] [data-slot="select-value"] option {
            color: #1f2937 !important;
        }

        /* Text inputs: value color for readability inside sidebar */
        /* Input value color inside the sidebar (e.g., model name input) */
        [data-testid="stSidebar"] .stTextInput>div>div>input,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #1f2937 !important;
        }

        /* Placeholder color (medium-dark gray) inside the sidebar inputs */
        [data-testid="stSidebar"] .stTextInput>div>div>input::placeholder,
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: #6b7280 !important;
            opacity: 1 !important;
        }

        /* Also ensure select placeholders (if present as first option) appear medium-dark */
        [data-testid="stSidebar"] .stSelectbox>div>div>select option:first-child,
        [data-testid="stSidebar"] [data-slot="select-value"] option:first-child,
        [data-testid="stSidebar"] [data-baseweb="select"] option:first-child {
            color: #6b7280 !important;
        }

        /* Tailwind/shadcn-like muted class override inside sidebar */
        [data-testid="stSidebar"] .text-muted-foreground,
        [data-testid="stSidebar"] .text-muted {
            color: #6b7280 !important;
        }

        /* If an element looks disabled via muted color, ensure it's readable unless actually disabled */
        [data-testid="stSidebar"] [aria-disabled="false"] {
            color: inherit !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        claude_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
        ]
        selected_model = st.selectbox(
            "Claude Model",
            claude_models,
            index=claude_models.index(st.session_state.claude_model) if st.session_state.claude_model in claude_models else 0,
        )
        st.session_state.claude_model = selected_model

                # Visualization options (theme selection moved to top of sidebar)
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

    # End of sidebar controls

    # Helper: render a compact analysis view for a single DataFrame
    def analyze_dataframe(name, df_local):
        """Render hero chips, sample download and a compact overview for df_local."""
        if df_local is None:
            st.markdown(f"<div class='section-title'>No data for {name}</div>", unsafe_allow_html=True)
            return

        num_rows = int(df_local.shape[0]) if df_local is not None else 0
        num_cols = int(df_local.shape[1]) if df_local is not None else 0
        missing_cells = int(df_local.isna().sum().sum()) if df_local is not None else 0
        total_cells = num_rows * num_cols if df_local is not None else 0
        missing_pct = round((missing_cells / total_cells * 100), 2) if total_cells else 0.0
        numeric_cols_list = df_local.select_dtypes('number').columns.tolist() if df_local is not None else []
        numeric_count = len(numeric_cols_list)
        if df_local is not None and total_cells:
            density_score = max(0.0, 100.0 - missing_pct)
            width_score = min(100.0, 20.0 + num_cols * 2.0)
            quality_score = int(round((density_score * 0.7) + (width_score * 0.3)))
        else:
            quality_score = 0

        st.markdown("<div class='surface-panel'>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='fx-h3'>Analysis – {name}</h3>", unsafe_allow_html=True)
        # hero chips
        chip_html = f"""
            <div class="chip-row">
              <div class="chip"><strong>Rows</strong> {num_rows:,}</div>
              <div class="chip"><strong>Cols</strong> {num_cols:,}</div>
              <div class="chip"><strong>Numeric</strong> {numeric_count}</div>
              <div class="chip"><strong>Missing</strong> {missing_pct:.2f}%</div>
              <div class="chip"><strong>Quality</strong> {quality_score}/100</div>
            </div>"""
        st.markdown(f"<div class='hero-panel'><div class='inner'><div class='pill'>{name}</div>{chip_html}</div></div>", unsafe_allow_html=True)

        # sample download
        try:
            csv_sample = df_local.head(1000).to_csv(index=False)
            st.download_button(f"Export CSV sample — {name}", csv_sample, f"{name}_sample.csv", "text/csv")
        except Exception:
            pass

        # small overview table
        with st.container():
            st.markdown("<div class='section-title'>Profile snapshot</div>", unsafe_allow_html=True)
            st.dataframe(df_local.head(5), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ---- Load data ----
df = None
# If uploaded DataFrames were parsed in the sidebar, use the selected active file
if st.session_state.get('uploaded_files'):
    uploaded_dfs = st.session_state['uploaded_files']
    active_name = st.session_state.get('active_file') or list(uploaded_dfs.keys())[0]
    df = uploaded_dfs.get(active_name)
    st.session_state.df = df
else:
    # no uploaded files; fall back to sample datasets
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

# If multiple uploaded files exist, render analysis for each and skip the main single-file hero
skip_main_analysis = False
if st.session_state.get('uploaded_files'):
    skip_main_analysis = True
    # render each uploaded file's analysis compactly
    for _name, _df in st.session_state['uploaded_files'].items():
        analyze_dataframe(_name, _df)
    # ensure downstream code still has an active df set
    active = st.session_state.get('active_file') or list(st.session_state['uploaded_files'].keys())[0]
    st.session_state.df = st.session_state['uploaded_files'][active]
    df = st.session_state.df

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
plotly_template = 'plotly_white' if ( 'theme' in locals() and theme == 'Light') else 'plotly_dark'

# Load persisted chat history into session state if available
if 'messages' not in st.session_state:
    st.session_state.messages = load_chat_history()

if not skip_main_analysis:
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
                    sample_text = df.head(100).to_csv(index=False)
                    model = genai.GenerativeModel(st.session_state.model_name)
                    prompt = f"You are a data analyst. Here is a sample of the dataset:\n{sample_text}\nUser question: {question}"
                    resp = model.generate_content(prompt)
                    text = getattr(resp,'text',None) or str(resp)
                    st.markdown("<div class='section-title' style='margin-top:8px'>Response</div>", unsafe_allow_html=True)
                    st.info(text)
                except Exception as e:
                    st.error(f"LLM error: {e}")
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview","Visualize","Chat"])

# ----- Overview tab -----
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if df is None:
        st.info("No data loaded — upload a file or choose a sample from the sidebar.")
    else:
        top_kpi, top_gauge = st.columns([2.4, 1.1])
        with top_kpi:
            st.markdown("<div class='section-title'>Profile snapshot</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class='kpi-grid'>
                  <div class='kpi-card'><div class='kpi-value'>{num_rows:,}</div><div class='kpi-label'>Rows</div></div>
                  <div class='kpi-card'><div class='kpi-value'>{num_cols:,}</div><div class='kpi-label'>Columns</div></div>
                  <div class='kpi-card'><div class='kpi-value'>{numeric_count}</div><div class='kpi-label'>Numeric</div></div>
                  <div class='kpi-card'><div class='kpi-value'>{missing_cells}</div><div class='kpi-label'>Missing cells</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with top_gauge:
            st.markdown("<div class='section-title'>Quality score</div>", unsafe_allow_html=True)
            try:
                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=quality_score,
                        number={"suffix": "/100", "font": {"color": "#e5f0ff"}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#ff3355"},
                            "bgcolor": "rgba(15,23,42,0.0)",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 40], "color": "rgba(239,68,68,0.25)"},
                                {"range": [40, 70], "color": "rgba(234,179,8,0.25)"},
                                {"range": [70, 100], "color": "rgba(34,197,94,0.32)"},
                            ],
                        },
                    )
                )
                gauge_fig.update_layout(
                    margin=dict(t=10, b=0, l=10, r=10),
                    height=170,
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    template=plotly_template,
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            except Exception:
                st.markdown(f"<div class='kpi-card'><div class='kpi-value'>{quality_score}</div><div class='kpi-label'>Quality score</div></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        cols_overview = st.columns([1.1, 1.4])
        with cols_overview[0]:
            st.markdown("<div class='section-title'>Column types</div>", unsafe_allow_html=True)
            type_counts = df.dtypes.astype(str).value_counts().to_frame().reset_index()
            type_counts.columns = ['dtype','count']
            st.dataframe(type_counts, use_container_width=True)
        with cols_overview[1]:
            st.markdown("<div class='section-title'>Missingness map</div>", unsafe_allow_html=True)
            try:
                miss = df.isna().astype(int)
                fig = px.imshow(miss.T, color_continuous_scale=['#020617','#ff3355'], aspect='auto')
                fig.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10))
                try:
                    fig.update_layout(template=plotly_template)
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Missingness map not available for this dataset.")

        # Optional correlation heatmap for numeric-only overview
        if len(numeric_cols_list) >= 2:
            st.markdown("<div class='section-title' style='margin-top:10px'>Correlation heatmap</div>", unsafe_allow_html=True)
            try:
                corr = df[numeric_cols_list].corr()
                fig_corr = px.imshow(
                    corr,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                )
                fig_corr.update_layout(height=260, margin=dict(t=24, b=10, l=10, r=10))
                try:
                    fig_corr.update_layout(template=plotly_template)
                except Exception:
                    pass
                st.plotly_chart(fig_corr, use_container_width=True)
            except Exception:
                st.info("Correlation heatmap not available for this dataset.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Visualize tab -----
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if df is None:
        st.info("Load a dataset to visualize")
    else:
        # filtering controls
        st.markdown("<div class='small'>Filters</div>", unsafe_allow_html=True)
        with st.expander("Add filters", expanded=False):
            filters = {}
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    vals = st.multiselect(f"{col}", options=df[col].dropna().unique().tolist(), key=f"f_{col}")
                    if vals:
                        filters[col] = vals
                elif np.issubdtype(df[col].dtype, np.number):
                    lo, hi = float(np.nanmin(df[col])), float(np.nanmax(df[col]))
                    r = st.slider(f"{col} range", min_value=lo, max_value=hi, value=(lo,hi), key=f"r_{col}")
                    filters[col] = r

        # apply filters
        dff = df.copy()
        for k,v in filters.items():
            if isinstance(v, tuple) and len(v) == 2:
                dff = dff[(dff[k] >= v[0]) & (dff[k] <= v[1])]
            elif isinstance(v, list) and v:
                dff = dff[dff[k].isin(v)]

        st.markdown("<div class='section-title'>Choose a chart</div>", unsafe_allow_html=True)
        chart_type = st.selectbox("Chart type", ["Histogram","Scatter","Box","Bar","Line"], index=0)

        # Detect numeric columns and also attempt to find "numeric-like" columns (strings with commas, % signs, currency)
        numeric_cols = dff.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = dff.select_dtypes(include=['object','category']).columns.tolist()

        # Identify convertible columns (heuristic): if after cleaning common artifacts a column converts to numeric for >50% of non-null values
        convertible_cols = []
        for col in dff.columns:
            if col in numeric_cols:
                continue
            ser = dff[col].dropna().astype(str)
            if ser.empty:
                continue
            # remove common formatting characters
            cleaned = ser.str.replace(r"[%$,]", "", regex=True).str.replace(r"\s+", "", regex=True)
            # remove any stray non-numeric characters except . and -
            cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
            coerced = pd.to_numeric(cleaned, errors='coerce')
            if coerced.notna().sum() >= 5 and (coerced.notna().mean() >= 0.5):
                convertible_cols.append(col)

        # Merge numeric and convertible for plotting choices; mark convertible columns so UI can show they are "converted"
        numeric_display = numeric_cols + convertible_cols
        numeric_display_labels = {col: (f"{col} (converted)" if col in convertible_cols else col) for col in numeric_display}

        if chart_type == "Histogram":
            if not numeric_display:
                st.info("No numeric or convertible columns available for histogram.")
                col = None
            else:
                col = st.selectbox("Variable", numeric_display, format_func=lambda c: numeric_display_labels.get(c, c))
            bins = st.slider("Bins", 5, 100, 30)
            if col is not None:
                # if selected column was convertible, try to coerce
                if col in convertible_cols:
                    tmp = dff[col].astype(str).str.replace(r"[%$,]","", regex=True).str.replace(r"\s+","", regex=True)
                    tmp = tmp.str.replace(r"[^0-9.\-]","", regex=True)
                    dff['_hist_numeric'] = pd.to_numeric(tmp, errors='coerce')
                    fig = px.histogram(dff, x='_hist_numeric', nbins=bins, title=f"Distribution: {col}")
                else:
                    fig = px.histogram(dff, x=col, nbins=bins, title=f"Distribution: {col}")
            fig.update_traces(marker_color='#ff3355')
            try:
                fig.update_layout(template=plotly_template)
            except Exception:
                pass
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            if not numeric_display:
                st.info("No numeric or convertible columns available for scatter plots.")
            else:
                x = st.selectbox("X", numeric_display, index=0, format_func=lambda c: numeric_display_labels.get(c, c))
                y = st.selectbox("Y", numeric_display, index=1 if len(numeric_display)>1 else 0, format_func=lambda c: numeric_display_labels.get(c, c))
                color = st.selectbox("Color (optional)", [None]+categorical_cols)

                # Defensive check: ensure selected columns still exist in the filtered dataframe
                dplot = dff.copy()
                if x not in dplot.columns or y not in dplot.columns:
                    st.error("Selected X or Y column is not present in the current dataset. Try reloading the data or selecting different columns.")
                else:
                    # Try converting to numeric (will introduce NaN for non-convertible values)
                    dplot['_x_numeric'] = pd.to_numeric(dplot[x], errors='coerce')
                    dplot['_y_numeric'] = pd.to_numeric(dplot[y], errors='coerce')

                    # Check that there is numeric data to plot
                    valid_pts = dplot[['_x_numeric','_y_numeric']].dropna()
                    if valid_pts.empty:
                        st.error(f"Cannot create scatter: selected X ('{x}') and Y ('{y}') do not contain numeric data after conversion.")
                    else:
                        # If trendline requested, ensure statsmodels is available; otherwise skip trendline
                        trendline = 'ols'
                        try:
                            import statsmodels.api as sm  # noqa: F401
                        except Exception:
                            trendline = None
                            st.warning("statsmodels not found; regression trendline will be disabled. Install `statsmodels` to enable trendline regression.")

                        try:
                            # Only pass color if that column exists in the dataframe
                            color_arg = color if (color and color in dplot.columns) else None
                            fig = px.scatter(dplot, x='_x_numeric', y='_y_numeric', color=color_arg, trendline=trendline,
                                             labels={'_x_numeric': x, '_y_numeric': y}, title=f"{y} vs {x}")
                            fig.update_traces(marker=dict(color='#ff3355'))
                            if color_arg:
                                fig.update_layout(legend_title_text=color_arg)
                            try:
                                fig.update_layout(template=plotly_template)
                            except Exception:
                                pass
                            st.plotly_chart(fig, use_container_width=True)
                        except ValueError as ve:
                            st.error(f"Plotly error: {ve}")
                        except Exception as e:
                            st.error(f"Unexpected error creating scatter plot: {e}")

        elif chart_type == "Box":
            if not numeric_display:
                st.info("No numeric or convertible columns available for box plot.")
                y = None
            else:
                y = st.selectbox("Y", numeric_display, format_func=lambda c: numeric_display_labels.get(c, c))
            x = st.selectbox("Group by (optional)", [None]+categorical_cols)
            if y is not None:
                fig = px.box(dff, x=x, y=y if y not in convertible_cols else '_box_numeric') if x else px.box(dff, y=y if y not in convertible_cols else '_box_numeric')
            else:
                fig = None
            if fig is not None:
                try:
                    fig.update_layout(template=plotly_template)
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            if categorical_cols:
                cat = st.selectbox("Category", categorical_cols)
                topn = st.slider("Top N categories", 1, 20, 10)
                grp = dff[cat].value_counts().nlargest(topn).reset_index()
                grp.columns = [cat,'count']
                fig = px.bar(grp, x=cat, y='count', color=cat)
                fig.update_layout(showlegend=False)
                try:
                    fig.update_layout(template=plotly_template)
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available for bar chart")

        elif chart_type == "Line":
            # try to find a datetime
            dt_cols = [c for c in dff.columns if np.issubdtype(dff[c].dtype, np.datetime64)]
            if dt_cols:
                dt = st.selectbox("Date column", dt_cols)
                if not numeric_display:
                    st.info("No numeric or convertible columns available for line chart")
                else:
                    y = st.selectbox("Y", numeric_display, format_func=lambda c: numeric_display_labels.get(c, c))
                    if y in convertible_cols:
                        # coerce to numeric into column used for line
                        tmp = dff[y].astype(str).str.replace(r"[%$,]","", regex=True).str.replace(r"\s+","", regex=True)
                        tmp = tmp.str.replace(r"[^0-9.\-]","", regex=True)
                        dff['_line_numeric'] = pd.to_numeric(tmp, errors='coerce')
                        fig = px.line(dff.sort_values(dt), x=dt, y='_line_numeric')
                    else:
                        fig = px.line(dff.sort_values(dt), x=dt, y=y)
                try:
                    fig.update_layout(template=plotly_template)
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No datetime columns found for line chart")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Data sample</div>", unsafe_allow_html=True)
        st.dataframe(dff.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Chat tab -----
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Chat with LLM</div>", unsafe_allow_html=True)

    if not st.session_state.gemini_api_key and llm_provider == 'Gemini':
        st.info("Set Gemini API key in the sidebar to use Chat.")
    elif llm_provider == "Claude" and not st.session_state.claude_api_key:
        st.info("Set Claude API key in the sidebar to use Chat.")
    else:
        # Dynamic suggestions based on available dataset columns
        st.markdown("<div class='small' style='margin-bottom:8px'>Suggested prompts</div>", unsafe_allow_html=True)
        suggestions = ["Summarize the dataset", "Find missing values", "Recommend important features"]
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist() if df is not None else []
        categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist() if df is not None else []

        # dataset-driven suggestions
        for c in numeric_cols[:3]:
            suggestions.append(f"Show distribution of {c}")
        if len(numeric_cols) >= 2:
            suggestions.append(f"Compare {numeric_cols[0]} vs {numeric_cols[1]}")
        for c in categorical_cols[:2]:
            suggestions.append(f"Count categories of {c}")

        # Render suggestion buttons (wrap if many)
        max_cols = min(len(suggestions), 6) if suggestions else 1
        cols_chunks = [suggestions[i:i+max_cols] for i in range(0, len(suggestions), max_cols)]
        for chunk in cols_chunks:
            cols_s = st.columns(len(chunk))
            for i, s in enumerate(chunk):
                if cols_s[i].button(s, key=f"sugg_{chunk[0]}_{i}"):
                    handle_prompt_and_respond(s)

        # Avatar images used when history is displayed
        user_img = "https://api.dicebear.com/6.x/initials/svg?seed=You&background=%23ff3355"
        assistant_img = "https://api.dicebear.com/6.x/bottts/svg?seed=Assistant&background=%236c757d"

        # Suggestion popup toggle (shows suggestions above the input, not under messages)
        if 'show_suggestions' not in st.session_state:
            st.session_state.show_suggestions = False

        # Row with toggles: Show history (left) and Suggestions (right)
        if 'show_history' not in st.session_state:
            st.session_state.show_history = False
        # slightly wider left column so the 'History' checkbox label doesn't wrap
        scols = st.columns([0.12, 0.76, 0.12])
        # left: show/hide history checkbox
        with scols[0]:
            show_hist = st.checkbox('History', value=st.session_state.show_history, key='chk_history')
            st.session_state.show_history = show_hist
        # right: suggestions toggle
        with scols[2]:
            if st.button('💡', key='toggle_suggestions'):
                st.session_state.show_suggestions = not st.session_state.show_suggestions

        # If history toggle is enabled, render the chat history in a scrollable container
        if st.session_state.show_history:
            msgs = st.session_state.get('messages', []) or []
            # render messages in a scrollable area
            st.markdown("<div style='max-height:420px; overflow:auto; padding:8px; border-radius:8px'>", unsafe_allow_html=True)
            for idx, m in enumerate(msgs):
                role = m.get('role', 'user')
                content = m.get('content', '')
                ts = m.get('ts')
                ts_html = f"<div class='ts'>{ts}</div>" if ts else ''
                if role == 'user':
                    c1, c2 = st.columns([0.08, 0.92])
                    with c1:
                        st.image(user_img, width=40)
                    with c2:
                        st.markdown(f"<div class='bubble' style='background:linear-gradient(90deg,#ffecec,#e8f0fe);padding:10px;border-radius:12px'><div class='muted'><strong>You</strong></div>{ts_html}<div style='margin-top:6px'>{content}</div></div>", unsafe_allow_html=True)
                else:
                    c1, c2 = st.columns([0.08, 0.92])
                    with c1:
                        st.image(assistant_img, width=40)
                    with c2:
                        st.markdown(f"<div class='bubble' style='background:#F6F6F6;padding:10px;border-radius:12px'><div class='muted'><strong>Assistant</strong></div>{ts_html}<div style='margin-top:6px'>{content}</div></div>", unsafe_allow_html=True)
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

