"""
pages/2_📊_Dashboard.py
Data visualization page for JNJ Budget Analyst.
Reads the shared csv_full DataFrame from session state (set on the Chat page).
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import statsmodels

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="JNJ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared dark CSS (same design tokens) ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg-base:        #0a0c10;
    --bg-surface:     #10141c;
    --bg-elevated:    #161b26;
    --bg-border:      #1e2535;
    --accent-green:   #00e5a0;
    --accent-blue:    #3b82f6;
    --accent-amber:   #f59e0b;
    --accent-purple:  #a78bfa;
    --accent-pink:    #f472b6;
    --text-primary:   #e8edf5;
    --text-secondary: #8b96a8;
    --text-muted:     #4a5568;
    --danger:         #ef4444;
    --font-display:   'Syne', sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-display) !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 70% 35% at 5% 0%,  rgba(0,229,160,0.04) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 95% 100%, rgba(59,130,246,0.05) 0%, transparent 55%),
        var(--bg-base);
}
footer { visibility: hidden; }
[data-testid="stToolbar"]  { display: none; }
[data-testid="stDecoration"] { display: none; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

/* ── Header ── */
.dash-header {
    display: flex; align-items: baseline; gap: 1rem;
    margin-bottom: 1.75rem; padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--bg-border);
}
.dash-header .logo-mark {
    font-family: var(--font-mono); font-size: 0.65rem; font-weight: 500;
    color: var(--accent-green); letter-spacing: 0.2em; text-transform: uppercase;
    background: rgba(0,229,160,0.08); border: 1px solid rgba(0,229,160,0.2);
    padding: 3px 10px; border-radius: 3px;
}
.dash-header h1 {
    font-size: 1.35rem !important; font-weight: 700 !important;
    color: var(--text-primary) !important; letter-spacing: -0.02em;
    margin: 0 !important; padding: 0 !important;
}
.dash-header .subtitle {
    font-family: var(--font-mono); font-size: 0.7rem;
    color: var(--text-muted); margin-left: auto; letter-spacing: 0.05em;
}

/* ── KPI cards ── */
.kpi-grid { display: flex; gap: 0.875rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: var(--bg-surface); border: 1px solid var(--bg-border);
    border-radius: 10px; padding: 1rem 1.25rem;
    position: relative; overflow: hidden;
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-color, var(--accent-green));
}
.kpi-label {
    font-family: var(--font-mono); font-size: 0.62rem; font-weight: 500;
    color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-family: var(--font-display); font-size: 1.45rem; font-weight: 800;
    color: var(--text-primary); letter-spacing: -0.03em; line-height: 1;
}
.kpi-sub {
    font-family: var(--font-mono); font-size: 0.62rem;
    color: var(--text-muted); margin-top: 0.3rem;
}

/* ── Section label ── */
.section-label {
    font-family: var(--font-mono); font-size: 0.65rem; font-weight: 500;
    color: var(--text-muted); letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.6rem; margin-top: 1.25rem;
    border-left: 2px solid var(--accent-green); padding-left: 0.6rem;
}

/* ── Chart wrapper ── */
.chart-card {
    background: var(--bg-surface); border: 1px solid var(--bg-border);
    border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 1rem;
}

/* ── Controls ── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {
    font-family: var(--font-mono) !important; font-size: 0.68rem !important;
    color: var(--text-secondary) !important; letter-spacing: 0.06em;
    text-transform: uppercase !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-elevated) !important; border-color: var(--bg-border) !important;
    color: var(--text-primary) !important; border-radius: 7px !important;
    font-family: var(--font-display) !important; font-size: 0.85rem !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--bg-border) !important; border-radius: 8px !important; overflow: hidden;
}
hr { border-color: var(--bg-border) !important; margin: 1.25rem 0 !important; }

[data-testid="stAlert"] {
    border-radius: 8px !important; font-family: var(--font-mono) !important;
    font-size: 0.76rem !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg-surface) !important; border-radius: 8px !important;
    padding: 3px !important; gap: 2px !important; border: 1px solid var(--bg-border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-mono) !important; font-size: 0.72rem !important;
    color: var(--text-muted) !important; letter-spacing: 0.06em; text-transform: uppercase;
    border-radius: 6px !important; padding: 0.4rem 1rem !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: var(--bg-elevated) !important; color: var(--accent-green) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-border); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

[data-testid="column"] { padding: 0 0.6rem !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child { padding-right: 0 !important; }

.stCaption, [data-testid="stCaption"] {
    font-family: var(--font-mono) !important; font-size: 0.68rem !important;
    color: var(--text-muted) !important;
}

</style>
""", unsafe_allow_html=True)

# ── Plotly dark template ───────────────────────────────────────────────────────
PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor":  "rgba(0,0,0,0)",
        "font":          {"family": "Syne, sans-serif", "color": "#8b96a8", "size": 11},
        "title":         {"font": {"family": "Syne, sans-serif", "color": "#e8edf5", "size": 14}, "x": 0.02},
        "xaxis": {
            "gridcolor": "#1e2535", "linecolor": "#1e2535",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 10},
            "title": {"font": {"color": "#4a5568"}},
            "zerolinecolor": "#1e2535",
        },
        "yaxis": {
            "gridcolor": "#1e2535", "linecolor": "#1e2535",
            "tickfont": {"family": "JetBrains Mono, monospace", "size": 10},
            "title": {"font": {"color": "#4a5568"}},
            "zerolinecolor": "#1e2535",
        },
        "legend": {
            "bgcolor": "rgba(16,20,28,0.8)", "bordercolor": "#1e2535", "borderwidth": 1,
            "font": {"family": "JetBrains Mono, monospace", "size": 10, "color": "#8b96a8"},
        },
        "colorway": ["#00e5a0", "#3b82f6", "#f59e0b", "#a78bfa", "#f472b6", "#34d399", "#60a5fa"],
        "margin": {"l": 48, "r": 20, "t": 48, "b": 48},
        "hoverlabel": {
            "bgcolor": "#161b26", "bordercolor": "#1e2535",
            "font": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#e8edf5"},
        },
    }
}

ACCENT_SEQ = ["#00e5a0", "#3b82f6", "#f59e0b", "#a78bfa", "#f472b6", "#34d399", "#60a5fa", "#fb923c"]

def apply_template(fig, height=380):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=height)
    return fig

def fmt_number(n):
    if abs(n) >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.2f}"

# ── Column classifier ──────────────────────────────────────────────────────────
def classify_columns(df: pd.DataFrame):
    numeric, date, categorical, text = [], [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date.append(col)
        else:
            # Try parse as date
            sample = df[col].dropna().astype(str).head(50)
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                    date.append(col)
                    continue
            except Exception:
                pass
            nuniq = df[col].nunique()
            if nuniq <= max(20, len(df) * 0.05):
                categorical.append(col)
            else:
                text.append(col)
    return numeric, date, categorical, text

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
    <span class="logo-mark">JNJ</span>
    <h1>Data Dashboard</h1>
    <span class="subtitle">visual analytics</span>
</div>
""", unsafe_allow_html=True)

# ── Guard: need data ──────────────────────────────────────────────────────────
df: pd.DataFrame | None = st.session_state.get("csv_full", None)

if df is None:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:3rem;margin-bottom:1rem;">📂</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                    color:#e8edf5;margin-bottom:0.5rem;">No data loaded</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#4a5568;">
            Upload a CSV on the Chat page, then come back here to explore it visually.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

numeric_cols, date_cols, cat_cols, text_cols = classify_columns(df)

# ── KPI strip ─────────────────────────────────────────────────────────────────
accent_cycle = ["var(--accent-green)", "var(--accent-blue)", "var(--accent-amber)", "var(--accent-purple)"]
kpi_items = [
    ("Rows", f"{len(df):,}", f"{df.shape[1]} columns"),
    ("Numeric cols", str(len(numeric_cols)), "quantitative fields"),
    ("Date cols", str(len(date_cols)), "time-series fields"),
    ("Category cols", str(len(cat_cols)), "categorical fields"),
]
if numeric_cols:
    first_num = numeric_cols[0]
    kpi_items.append((f"Σ {first_num}", fmt_number(df[first_num].sum()), f"avg {fmt_number(df[first_num].mean())}"))

cards_html = '<div class="kpi-grid">'
for i, (label, value, sub) in enumerate(kpi_items):
    ac = accent_cycle[i % len(accent_cycle)]
    cards_html += f"""
    <div class="kpi-card" style="--accent-color:{ac};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>"""
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_time, tab_dist, tab_compare, tab_table = st.tabs([
    "📈 Overview", "🕐 Time Series", "📊 Distributions", "🔀 Compare", "🗂 Full Table"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-label">Summary statistics</div>', unsafe_allow_html=True)

    if not numeric_cols:
        st.info("No numeric columns detected for summary stats.")
    else:
        summary = df[numeric_cols].describe().T.reset_index()
        summary.columns = ["Column", "Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]
        for col in ["Count", "Mean", "Std", "Min", "25%", "Median", "75%", "Max"]:
            summary[col] = summary[col].apply(lambda x: f"{x:,.2f}")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    if len(numeric_cols) >= 2:
        st.markdown('<div class="section-label">Correlation heatmap</div>', unsafe_allow_html=True)
        corr = df[numeric_cols].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values.tolist(),
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale=[[0, "#3b82f6"], [0.5, "#161b26"], [1, "#00e5a0"]],
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
        ))
        apply_template(fig_corr, height=400)
        st.plotly_chart(fig_corr, use_container_width=True)

    if cat_cols and numeric_cols:
        st.markdown('<div class="section-label">Category breakdown</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        cat_col = c1.selectbox("Category column", cat_cols, key="ov_cat")
        num_col = c2.selectbox("Numeric column", numeric_cols, key="ov_num")
        agg_fn  = c1.selectbox("Aggregate by", ["sum", "mean", "median", "count"], key="ov_agg")

        agg_df = df.groupby(cat_col)[num_col].agg(agg_fn).reset_index().sort_values(num_col, ascending=False).head(25)
        fig_bar = px.bar(
            agg_df, x=cat_col, y=num_col,
            color_discrete_sequence=[ACCENT_SEQ[0]],
            labels={num_col: f"{agg_fn}({num_col})", cat_col: ""},
        )
        fig_bar.update_traces(marker_line_width=0)
        apply_template(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_time:
    if not date_cols:
        st.info("No date/datetime columns were detected in your data. Try casting a column to datetime on the Chat page.")
    else:
        c1, c2, c3 = st.columns(3)
        date_col  = c1.selectbox("Date column", date_cols, key="ts_date")
        value_col = c2.selectbox("Value column", numeric_cols, key="ts_val") if numeric_cols else None
        group_col = c3.selectbox("Group by (optional)", ["— none —"] + cat_cols, key="ts_grp")
        agg_period = c1.selectbox("Resample", ["None", "D — daily", "W — weekly", "M — monthly", "Q — quarterly", "Y — yearly"], key="ts_res")
        chart_type = c2.selectbox("Chart type", ["Line", "Area", "Bar"], key="ts_chart")

        if value_col:
            ts_df = df[[date_col, value_col] + ([group_col] if group_col != "— none —" else [])].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df = ts_df.dropna(subset=[date_col])

            period_map = {"D — daily": "D", "W — weekly": "W", "M — monthly": "ME", "Q — quarterly": "QE", "Y — yearly": "YE"}
            period = period_map.get(agg_period)

            if period and group_col == "— none —":
                ts_df = ts_df.set_index(date_col).resample(period)[value_col].sum().reset_index()
            elif period and group_col != "— none —":
                ts_df = (ts_df.groupby([pd.Grouper(key=date_col, freq=period), group_col])[value_col]
                         .sum().reset_index())

            color_arg = group_col if group_col != "— none —" else None
            color_seq = ACCENT_SEQ

            if chart_type == "Line":
                fig_ts = px.line(ts_df, x=date_col, y=value_col, color=color_arg, color_discrete_sequence=color_seq)
                fig_ts.update_traces(line_width=2)
            elif chart_type == "Area":
                fig_ts = px.area(ts_df, x=date_col, y=value_col, color=color_arg, color_discrete_sequence=color_seq)
                fig_ts.update_traces(line_width=1.5)
            else:
                fig_ts = px.bar(ts_df, x=date_col, y=value_col, color=color_arg, color_discrete_sequence=color_seq)
                fig_ts.update_traces(marker_line_width=0)

            apply_template(fig_ts, height=420)
            st.plotly_chart(fig_ts, use_container_width=True)

            # Rolling average overlay for line/area
            if chart_type in ("Line", "Area") and color_arg is None and period in (None, "D — daily"):
                window = st.slider("Rolling average window (rows)", 2, min(90, len(ts_df)//2 or 2), 7, key="ts_roll")
                ts_df["_roll"] = ts_df[value_col].rolling(window, min_periods=1).mean()
                fig_roll = go.Figure()
                fig_roll.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df[value_col],
                                              mode="lines", name=value_col,
                                              line=dict(color="#1e2535", width=1)))
                fig_roll.add_trace(go.Scatter(x=ts_df[date_col], y=ts_df["_roll"],
                                              mode="lines", name=f"{window}-period avg",
                                              line=dict(color="#00e5a0", width=2.5)))
                apply_template(fig_roll, height=340)
                st.plotly_chart(fig_roll, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_dist:
    if not numeric_cols:
        st.info("No numeric columns to display distributions for.")
    else:
        c1, c2 = st.columns(2)
        num_col = c1.selectbox("Numeric column", numeric_cols, key="dist_num")
        color_by = c2.selectbox("Color by (optional)", ["— none —"] + cat_cols, key="dist_col")
        chart_mode = c1.selectbox("Chart", ["Histogram", "Box plot", "Violin", "ECDF"], key="dist_mode")

        color_arg = color_by if color_by != "— none —" else None

        if chart_mode == "Histogram":
            nbins = st.slider("Bins", 5, 100, 30, key="dist_bins")
            fig_d = px.histogram(df, x=num_col, color=color_arg, nbins=nbins,
                                 color_discrete_sequence=ACCENT_SEQ, opacity=0.85,
                                 marginal="rug")
            fig_d.update_traces(marker_line_width=0)
        elif chart_mode == "Box plot":
            fig_d = px.box(df, y=num_col, color=color_arg,
                           color_discrete_sequence=ACCENT_SEQ, points="outliers")
        elif chart_mode == "Violin":
            fig_d = px.violin(df, y=num_col, color=color_arg,
                              color_discrete_sequence=ACCENT_SEQ, box=True, points=False)
        else:  # ECDF
            fig_d = px.ecdf(df, x=num_col, color=color_arg, color_discrete_sequence=ACCENT_SEQ)

        apply_template(fig_d, height=420)
        st.plotly_chart(fig_d, use_container_width=True)

        # Mini grid of histograms for all numeric cols
        if len(numeric_cols) > 1:
            st.markdown('<div class="section-label">All numeric columns — distribution overview</div>', unsafe_allow_html=True)
            n = len(numeric_cols)
            cols_per_row = 3
            rows_needed = (n + cols_per_row - 1) // cols_per_row
            sub_fig = make_subplots(rows=rows_needed, cols=cols_per_row,
                                    subplot_titles=numeric_cols,
                                    vertical_spacing=0.12, horizontal_spacing=0.08)
            for idx, col in enumerate(numeric_cols):
                r, c = divmod(idx, cols_per_row)
                vals = df[col].dropna()
                sub_fig.add_trace(
                    go.Histogram(x=vals, nbinsx=25, marker_color=ACCENT_SEQ[idx % len(ACCENT_SEQ)],
                                 marker_line_width=0, showlegend=False, name=col),
                    row=r + 1, col=c + 1,
                )
            apply_template(sub_fig, height=max(280, rows_needed * 210))
            sub_fig.update_annotations(font=dict(family="JetBrains Mono, monospace", size=10, color="#8b96a8"))
            st.plotly_chart(sub_fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns to compare.")
    else:
        c1, c2, c3 = st.columns(3)
        x_col    = c1.selectbox("X axis", numeric_cols, key="cmp_x")
        y_col    = c2.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="cmp_y")
        color_c  = c3.selectbox("Color by", ["— none —"] + cat_cols, key="cmp_col")
        size_c   = c1.selectbox("Size by (bubble)", ["— none —"] + numeric_cols, key="cmp_sz")
        mode     = c2.selectbox("Chart type", ["Scatter", "Bubble", "Scatter matrix"], key="cmp_mode")

        color_arg = color_c if color_c != "— none —" else None
        size_arg  = size_c  if size_c  != "— none —" else None

        plot_df = df.dropna(subset=[x_col, y_col])

        if mode == "Scatter":
            fig_c = px.scatter(plot_df, x=x_col, y=y_col, color=color_arg,
                               color_discrete_sequence=ACCENT_SEQ,
                               trendline="ols", trendline_color_override="#f59e0b",
                               opacity=0.7)
        elif mode == "Bubble":
            fig_c = px.scatter(plot_df, x=x_col, y=y_col, color=color_arg, size=size_arg,
                               color_discrete_sequence=ACCENT_SEQ,
                               size_max=30, opacity=0.7)
        else:
            dims = st.multiselect("Dimensions", numeric_cols, default=numeric_cols[:min(4, len(numeric_cols))], key="cmp_dims")
            if dims:
                fig_c = px.scatter_matrix(plot_df, dimensions=dims, color=color_arg,
                                          color_discrete_sequence=ACCENT_SEQ, opacity=0.6)
                fig_c.update_traces(diagonal_visible=False, marker_size=3)
                apply_template(fig_c, height=560)
                st.plotly_chart(fig_c, use_container_width=True)
            else:
                st.info("Select at least 2 dimensions.")
            fig_c = None

        if fig_c is not None:
            fig_c.update_traces(marker_line_width=0)
            apply_template(fig_c, height=440)
            st.plotly_chart(fig_c, use_container_width=True)

        # Bar comparison of multiple numeric cols
        if cat_cols:
            st.markdown('<div class="section-label">Multi-metric bar comparison</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            group_c  = c1.selectbox("Group by", cat_cols, key="bar_grp")
            metrics  = c2.multiselect("Metrics", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="bar_met")
            if metrics:
                agg = df.groupby(group_c)[metrics].mean().reset_index()
                fig_multi = px.bar(agg, x=group_c, y=metrics, barmode="group",
                                   color_discrete_sequence=ACCENT_SEQ)
                fig_multi.update_traces(marker_line_width=0)
                apply_template(fig_multi, height=380)
                st.plotly_chart(fig_multi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FULL TABLE
# ══════════════════════════════════════════════════════════════════════════════
with tab_table:
    st.markdown('<div class="section-label">Filter & explore</div>', unsafe_allow_html=True)

    # Column picker
    selected_cols = st.multiselect(
        "Columns to show",
        df.columns.tolist(),
        default=df.columns.tolist()[:min(10, len(df.columns))],
        key="tbl_cols",
    )

    view_df = df[selected_cols] if selected_cols else df

    # Filter on a categorical column
    if cat_cols:
        c1, c2 = st.columns(2)
        filter_col = c1.selectbox("Filter column", ["— none —"] + [c for c in cat_cols if c in view_df.columns], key="tbl_fc")
        if filter_col != "— none —":
            options = ["All"] + sorted(view_df[filter_col].dropna().unique().tolist())
            filter_val = c2.selectbox("Filter value", options, key="tbl_fv")
            if filter_val != "All":
                view_df = view_df[view_df[filter_col] == filter_val]

    # Sort
    c1, c2 = st.columns(2)
    sort_col = c1.selectbox("Sort by", ["— none —"] + view_df.columns.tolist(), key="tbl_sc")
    sort_asc = c2.selectbox("Order", ["Ascending", "Descending"], key="tbl_so") == "Ascending"
    if sort_col != "— none —":
        view_df = view_df.sort_values(sort_col, ascending=sort_asc)

    st.caption(f"SHOWING {len(view_df):,} OF {len(df):,} ROWS · {len(view_df.columns)} COLUMNS")
    st.dataframe(view_df, use_container_width=True, height=520)

    # Download button
    csv_bytes = view_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download filtered CSV",
        data=csv_bytes,
        file_name="jnj_filtered_export.csv",
        mime="text/csv",
    )

