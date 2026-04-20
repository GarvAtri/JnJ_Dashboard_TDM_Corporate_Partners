import streamlit as st
import pandas as pd
import numpy as np
import os
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Page configuration
st.set_page_config(page_title="LLM Data Studio", layout="wide", initial_sidebar_state="expanded")

_CSS = """
/***** Modern card-like layout and subtle shadows *****/
:root{--accent:#E31937;--muted:#6c757d;--bg:#ffffff}
body{background:var(--bg)}
.card{background:#fff;border-radius:12px;padding:18px;box-shadow:0 6px 20px rgba(0,0,0,0.06);}
.small{font-size:13px;color:var(--muted)}
.kpi{font-size:22px;font-weight:700;color:var(--accent)}
.section-title{font-size:18px;color:var(--accent);font-weight:700}
.ghost-btn>button{background:transparent;border:1px solid var(--accent);color:var(--accent);}
/* make Streamlit inputs a bit larger and rounded */
.stTextInput>div>div>input, .stSelectbox>div>div>select {border-radius:8px;padding:10px}
"""

st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)

# ---- Sidebar: controls ----
with st.sidebar:
    st.markdown("""<div style='text-align:center'>
        <h2 style='color:#E31937;margin:6px 0'>LLM Data Studio</h2>
        <div class='small'>A modern interactive dashboard — upload data, visualize, and ask the LLM.</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # Upload and sample dataset shortcuts
    uploaded_file = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"], help="Drag & drop a file or click to browse")
    st.markdown("---")
    st.markdown("<div class='small'>Or try a sample dataset:</div>", unsafe_allow_html=True)
    sample = st.selectbox("Load sample", ["—","Iris","Tips (seaborn)","Wine (sklearn)"])

    st.markdown("---")
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

    st.markdown("---")
    st.markdown("<div class='small'>Visualization options</div>", unsafe_allow_html=True)
    theme = st.selectbox("Theme", ["Light","Dark"], index=0)
    st.markdown("---")
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            if k not in ["gemini_api_key","model_name"]:
                del st.session_state[k]
        st.experimental_rerun()

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

# ---- Top hero / tabs ----
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<div style='display:flex;align-items:center;gap:16px'>", unsafe_allow_html=True)
    st.markdown("<div><h1 style='margin:0;color:#E31937'>LLM Data Studio</h1><div class='small'>A clean, modern dashboard for data + LLM analysis</div></div>", unsafe_allow_html=True)
with col2:
    if df is not None:
        csv = df.head(1000).to_csv(index=False)
        st.download_button("Download sample CSV", csv, "sample.csv", "text/csv")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

tabs = st.tabs(["Overview","Visualize","Analyze","Chat"])

# ----- Overview tab -----
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if df is None:
        st.info("No data loaded — upload a file or choose a sample from the sidebar.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric cols", len(df.select_dtypes('number').columns))
        c4.metric("Missing cells", int(df.isna().sum().sum()))

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Column types</div>", unsafe_allow_html=True)
        type_counts = df.dtypes.astype(str).value_counts().to_frame().reset_index()
        type_counts.columns = ['dtype','count']
        st.dataframe(type_counts, use_container_width=True)

        st.markdown("<div class='section-title' style='margin-top:14px'>Missingness map</div>", unsafe_allow_html=True)
        try:
            miss = df.isna().astype(int)
            fig = px.imshow(miss.T, color_continuous_scale=['#ffffff','#E31937'], aspect='auto')
            fig.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.info("Missingness map not available for this dataset.")
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

        numeric_cols = dff.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = dff.select_dtypes(include=['object','category']).columns.tolist()

        if chart_type == "Histogram":
            col = st.selectbox("Variable", numeric_cols)
            bins = st.slider("Bins", 5, 100, 30)
            fig = px.histogram(dff, x=col, nbins=bins, title=f"Distribution: {col}")
            fig.update_traces(marker_color='#E31937')
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Scatter":
            x = st.selectbox("X", numeric_cols, index=0)
            y = st.selectbox("Y", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
            color = st.selectbox("Color (optional)", [None]+categorical_cols)
            fig = px.scatter(dff, x=x, y=y, color=color, trendline='ols')
            fig.update_traces(marker=dict(color='#E31937'))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box":
            y = st.selectbox("Y", numeric_cols)
            x = st.selectbox("Group by (optional)", [None]+categorical_cols)
            fig = px.box(dff, x=x, y=y) if x else px.box(dff, y=y)
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Bar":
            if categorical_cols:
                cat = st.selectbox("Category", categorical_cols)
                topn = st.slider("Top N categories", 1, 20, 10)
                grp = dff[cat].value_counts().nlargest(topn).reset_index()
                grp.columns = [cat,'count']
                fig = px.bar(grp, x=cat, y='count', color=cat)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available for bar chart")

        elif chart_type == "Line":
            # try to find a datetime
            dt_cols = [c for c in dff.columns if np.issubdtype(dff[c].dtype, np.datetime64)]
            if dt_cols:
                dt = st.selectbox("Date column", dt_cols)
                y = st.selectbox("Y", numeric_cols)
                fig = px.line(dff.sort_values(dt), x=dt, y=y)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No datetime columns found for line chart")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Data sample</div>", unsafe_allow_html=True)
        st.dataframe(dff.head(200), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Analyze tab (LLM) -----
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    if df is None:
        st.info("Upload data or choose a sample to analyze with the LLM.")
    else:
        st.markdown("<div class='section-title'>LLM Analysis</div>", unsafe_allow_html=True)
        question = st.text_area("Ask the LLM about this data", placeholder="Ask about trends, anomalies, summaries, feature importance...")
        if st.button("Analyze"):
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
                        st.markdown("<div class='small'>LLM Response</div>", unsafe_allow_html=True)
                        st.info(text)
                    except Exception as e:
                        st.error(f"LLM error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Chat tab -----
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Chat with LLM</div>", unsafe_allow_html=True)
    if not st.session_state.gemini_api_key:
        st.info("Set Gemini API key in the sidebar to use Chat.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for m in st.session_state.messages:
            with st.chat_message(m['role']):
                st.markdown(m['content'])

        if prompt := st.chat_input("Start conversation..."):
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role':'user','content':prompt})
            if not st.session_state.model_name:
                st.error("Select a model in the sidebar first.")
            else:
                try:
                    model = genai.GenerativeModel(st.session_state.model_name)
                    ctx = ''
                    if df is not None:
                        ctx = f"\nContext: columns = {list(df.columns)}"
                    resp = model.generate_content(prompt + ctx)
                    text = getattr(resp,'text',None) or str(resp)
                    with st.chat_message('assistant'):
                        st.markdown(text)
                    st.session_state.messages.append({'role':'assistant','content':text})
                except Exception as e:
                    st.error(f"Chat error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

