import os
import io
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from google import genai
from google.genai.types import GenerateContentConfig

# ---------- Setup ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="JNJ Budget Analyst",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Dark Theme CSS (Bad) ----------
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg-base:       #0a0c10;
    --bg-surface:    #10141c;
    --bg-elevated:   #161b26;
    --bg-border:     #1e2535;
    --accent-green:  #00e5a0;
    --accent-blue:   #3b82f6;
    --accent-amber:  #f59e0b;
    --text-primary:  #e8edf5;
    --text-secondary:#8b96a8;
    --text-muted:    #4a5568;
    --danger:        #ef4444;
    --font-display:  'Syne', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 40% at 10% 0%, rgba(0,229,160,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(59,130,246,0.05) 0%, transparent 60%),
        var(--bg-base);
}
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

.app-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin-bottom: 2rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--bg-border);
}
.app-header .logo-mark {
    font-family: var(--font-mono); font-size: 0.65rem; font-weight: 500;
    color: var(--accent-green); letter-spacing: 0.2em; text-transform: uppercase;
    background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.2);
    padding: 3px 10px; border-radius: 3px;
}
.app-header h1 {
    font-size: 1.35rem !important; font-weight: 700 !important;
    color: var(--text-primary) !important; letter-spacing: -0.02em;
    margin: 0 !important; padding: 0 !important;
}
.app-header .subtitle {
    font-family: var(--font-mono); font-size: 0.7rem;
    color: var(--text-muted); margin-left: auto; letter-spacing: 0.05em;
}

h2, h3 {
    font-family: var(--font-display) !important; font-weight: 700 !important;
    letter-spacing: -0.02em !important; color: var(--text-primary) !important;
}
h2 { font-size: 1rem !important; margin-bottom: 1rem !important; }
h3 { font-size: 0.8rem !important; }

[data-testid="stChatMessageContainer"] { background: transparent !important; }
[data-testid="stChatMessage"] {
    background: var(--bg-surface) !important; border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important; padding: 1rem 1.25rem !important;
    margin-bottom: 0.6rem !important; font-family: var(--font-display) !important;
    font-size: 0.9rem !important; color: var(--text-primary) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-color: rgba(0,229,160,0.2) !important;
    background: rgba(0,229,160,0.03) !important;
}
[data-testid="chatAvatarIcon-user"] svg,
[data-testid="chatAvatarIcon-assistant"] svg { color: var(--accent-green) !important; }

[data-testid="stChatInput"] {
    background: var(--bg-elevated) !important; border: 1px solid var(--bg-border) !important;
    border-radius: 10px !important; padding: 0.1rem 0.5rem !important; margin-top: 0.5rem;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent-green) !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,0.12) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-display) !important; font-size: 0.875rem !important;
    color: var(--text-primary) !important; background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }

[data-testid="stFileUploader"] {
    border: 1px dashed var(--bg-border) !important; border-radius: 8px !important;
    background: var(--bg-elevated) !important; padding: 0.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent-green) !important; }
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important; font-size: 0.78rem !important;
    font-family: var(--font-mono) !important;
}

[data-testid="stSlider"] label {
    font-family: var(--font-mono) !important; font-size: 0.72rem !important;
    color: var(--text-secondary) !important; letter-spacing: 0.05em; text-transform: uppercase;
}
[data-testid="stSlider"] [role="slider"] {
    background: var(--accent-green) !important; border-color: var(--accent-green) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--bg-border) !important; border-radius: 8px !important; overflow: hidden;
}

.stButton > button {
    font-family: var(--font-mono) !important; font-size: 0.72rem !important;
    font-weight: 500 !important; letter-spacing: 0.08em !important;
    text-transform: uppercase !important; color: var(--text-secondary) !important;
    background: var(--bg-elevated) !important; border: 1px solid var(--bg-border) !important;
    border-radius: 6px !important; padding: 0.5rem 1rem !important;
    width: 100% !important; transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: var(--danger) !important; color: var(--danger) !important;
    background: rgba(239,68,68,0.06) !important;
}

hr { border-color: var(--bg-border) !important; margin: 1.25rem 0 !important; }

[data-testid="stAlert"] {
    border-radius: 8px !important; font-family: var(--font-mono) !important;
    font-size: 0.76rem !important; border-left-width: 3px !important;
}

.stCaption, [data-testid="stCaption"] {
    font-family: var(--font-mono) !important; font-size: 0.68rem !important;
    color: var(--text-muted) !important; letter-spacing: 0.03em;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--bg-border) !important;
}

[data-testid="column"] { padding: 0 0.75rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

[data-testid="stMarkdownContainer"] p {
    font-size: 0.9rem !important; line-height: 1.7 !important; color: var(--text-primary) !important;
}
[data-testid="stMarkdownContainer"] code {
    font-family: var(--font-mono) !important; font-size: 0.78rem !important;
    background: rgba(59,130,246,0.1) !important; color: #93c5fd !important;
    padding: 1px 5px !important; border-radius: 3px !important;
}
[data-testid="stMarkdownContainer"] pre {
    background: var(--bg-base) !important; border: 1px solid var(--bg-border) !important;
    border-radius: 8px !important; padding: 1rem !important;
}
[data-testid="stMarkdownContainer"] pre code {
    background: transparent !important; color: var(--text-secondary) !important;
}

</style>
"""

# st.markdown(DARK_CSS, unsafe_allow_html=True)

if not api_key:
    st.error("Missing GEMINI_API_KEY environment variable. Add it to your .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

PERSONA = (
    "You are an expert AI assistant for statisticians and data analysts working with financial data. "
    "You are knowledgeable about statistical methods, financial modeling, and data analysis techniques. "
    "You provide explanations, code examples, and guidance on various topics related to financial data analysis. "
    "Detect csv file formatting, and respond accordingly."
)

# ---------- Helpers ----------
def build_prompt(persona_text: str, turns, user_msg: str, max_turns: int = 8) -> str:
    turns = turns[-max_turns:]
    lines = [persona_text, "", "Conversation:"]
    for u, a in turns:
        lines.append(f"User: {u}")
        lines.append(f"AI: {a}")
    lines.append(f"User: {user_msg}")
    lines.append("AI:")
    return "\n".join(lines)

def sniff_csv(bytes_data: bytes, sample_chars: int = 50_000):
    text = bytes_data[:sample_chars].decode("utf-8", errors="replace")
    candidates = [",", "\t", ";", "|"]
    best = None
    for sep in candidates:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, nrows=50)
            score = (df.shape[1], -sum(c.startswith("Unnamed") for c in df.columns))
            if best is None or score > best["score"]:
                best = {"sep": sep, "df": df, "score": score}
        except Exception:
            continue
    if best is None:
        return None, None
    return best["sep"], best["df"]

def load_full_csv(bytes_data: bytes, sep: str) -> pd.DataFrame:
    text = bytes_data.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text), sep=sep)

def format_csv_context(sep, df_preview):
    if sep is None or df_preview is None:
        return "No CSV detected or parsing failed."
    cols = list(df_preview.columns)
    sample_rows = df_preview.head(8).to_dict(orient="records")
    return (
        "CSV detected.\n"
        f"Likely delimiter: {repr(sep)}\n"
        f"Columns ({len(cols)}): {cols}\n"
        f"Sample rows (first {min(len(sample_rows), 8)}): {sample_rows}\n"
        "If the user asks about formatting/cleaning, propose parsing settings and cleaning steps."
    )

# ---------- Session State ----------
for key, default in [
    ("history", []),
    ("messages", []),
    ("csv_bytes", None),
    ("csv_sep", None),
    ("csv_preview", None),
    ("csv_full", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------- Header ----------
st.markdown("""
<div class="app-header">
    <span class="logo-mark">JNJ</span>
    <h1>Budget Dashboard Analyst</h1>
    <span class="subtitle">powered by Gemini 2.5 Flash</span>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);margin-bottom:1.5rem;">'
    '→ Switch to <b style="color:var(--accent-green)">📊 Dashboard</b> in the sidebar to visualize your data</p>',
    unsafe_allow_html=True,
)

# ---------- Layout ----------
left, right = st.columns([2, 1], gap="large")

with right:
    st.markdown("#### 📂 Data Source")
    uploaded = st.file_uploader(
        "Upload a CSV or TSV file",
        type=["csv", "txt"],
        label_visibility="collapsed",
        help="Supports comma, tab, semicolon, and pipe-delimited files",
    )
    if uploaded is not None:
        st.session_state.csv_bytes = uploaded.read()
        sep, preview = sniff_csv(st.session_state.csv_bytes)
        st.session_state.csv_sep = sep
        st.session_state.csv_preview = preview
        if sep is not None:
            st.session_state.csv_full = load_full_csv(st.session_state.csv_bytes, sep)

    if st.session_state.csv_preview is not None:
        delim_label = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}.get(
            st.session_state.csv_sep, repr(st.session_state.csv_sep)
        )
        rows = len(st.session_state.csv_full) if st.session_state.csv_full is not None else "?"
        st.success(f"✓ Parsed — {delim_label} · {st.session_state.csv_preview.shape[1]} cols · {rows} rows")
        st.caption("PREVIEW — FIRST 50 ROWS")
        st.dataframe(st.session_state.csv_preview, use_container_width=True, height=220)
    elif st.session_state.csv_bytes is not None:
        st.warning("⚠ Could not detect a valid delimiter in the uploaded file.")

    st.divider()
    st.markdown("#### ⚙ Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1,
                            help="Higher = more creative; lower = more deterministic")
    max_tokens = st.slider("Max Output Tokens", 128, 4096, 1024, 128)

    st.divider()
    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(
        f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:0.5rem;">'
        f'SESSION · {msg_count} message{"s" if msg_count != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    if st.button("↺ Clear Chat"):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

with left:
    st.markdown("#### 💬 Chat")

    if not st.session_state.messages:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;color:var(--text-muted);'
            'font-family:var(--font-mono);font-size:0.75rem;letter-spacing:0.05em;">'
            '— Start a conversation or upload a CSV to get started —</div>',
            unsafe_allow_html=True,
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about your financial data, statistical methods, or uploaded CSV…")

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        csv_context = ""
        if st.session_state.csv_bytes is not None:
            csv_context = "\n\n[File Context]\n" + format_csv_context(
                st.session_state.csv_sep, st.session_state.csv_preview
            )

        prompt = build_prompt(PERSONA, st.session_state.history, user_msg) + csv_context

        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=GenerateContentConfig(
                            temperature=float(temperature),
                            max_output_tokens=int(max_tokens),
                        ),
                    )
                    ai_text = (resp.text or "").strip() or "[No text returned]"
                except Exception as e:
                    ai_text = f"Error: {e}"
            st.markdown(ai_text)

        st.session_state.messages.append({"role": "assistant", "content": ai_text})
        st.session_state.history.append((user_msg, ai_text))