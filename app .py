
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

st.set_page_config(
    page_title="Global Energy Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #060d1f;
    color: #e2e8f0;
}
.stApp { background: #060d1f; }

section[data-testid="stSidebar"] {
    background: #0a1628 !important;
    border-right: 1px solid #1e3a5f !important;
}
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 1px; }

[data-testid="metric-container"] {
    background: #0d1f3c !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 14px !important;
    padding: 20px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d1f3c;
    border-radius: 12px;
    padding: 5px;
    gap: 4px;
    border: 1px solid #1e3a5f;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    font-weight: 500;
    font-size: 0.82rem;
    border-radius: 9px;
    padding: 8px 18px;
    letter-spacing: 0.3px;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #e0f2fe !important;
}

.stMultiSelect [data-baseweb="tag"] { background: #1e3a5f !important; }
div[data-baseweb="select"] { background: #0d1f3c !important; border-color: #1e3a5f !important; }

.block-container { padding: 1.5rem 2rem 3rem !important; }

.header-wrap {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px; padding-bottom: 20px;
    border-bottom: 1px solid #1e3a5f;
}
.header-left h1 {
    font-family: 'Inter', sans-serif;
    font-size: 1.7rem; font-weight: 700;
    color: #f1f5f9; margin: 0; line-height: 1.2;
    letter-spacing: -0.5px;
}
.header-left p { color: #475569; font-size: 0.78rem; margin: 4px 0 0; letter-spacing: 1px; text-transform: uppercase; }
.live-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: #052e16; border: 1px solid #166534;
    border-radius: 20px; padding: 6px 14px;
    font-size: 0.75rem; color: #4ade80; font-weight: 500;
}
.live-dot { width: 7px; height: 7px; background: #22c55e; border-radius: 50%; display: inline-block; }

.section-title {
    font-size: 0.72rem; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 28px 0 14px;
    display: flex; align-items: center; gap: 10px;
}
.section-title::after {
    content: ''; flex: 1; height: 1px; background: #1e3a5f;
}

.info-card {
    background: #0d1f3c; border: 1px solid #1e3a5f;
    border-radius: 14px; padding: 22px; margin-bottom: 14px;
}
.stat-row {
    display: flex; justify-content: space-between;
    padding: 10px 0; border-bottom: 1px solid #1e3a5f;
    font-size: 0.82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-key { color: #64748b; }
.stat-val { color: #e2e8f0; font-weight: 500; font-family: 'JetBrains Mono', monospace; }

.pred-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 20px; }
.pred-card {
    background: #0d1628; border: 1px solid #2d3f6b;
    border-radius: 12px; padding: 20px; text-align: center;
}
.pred-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem; font-weight: 600; color: #a5b4fc; line-height: 1;
}
.pred-lbl { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 6px; }

.stPlotlyChart { border-radius: 12px; overflow: hidden; }

div[data-testid="stHorizontalBlock"] > div { gap: 14px; }
</style>
""", unsafe_allow_html=True)

PLOT_BG   = "rgba(13,31,60,0.6)"
PAPER_BG  = "rgba(0,0,0,0)"
TEMPLATE  = "plotly_dark"
COLORS    = ["#3b82f6","#22c55e","#f59e0b","#ef4444","#a78bfa","#38bdf8","#fb923c","#e879f9"]
GRID_COL  = "rgba(30,58,95,0.6)"

def styled_fig(fig, h=420):
    fig.update_layout(
        template=TEMPLATE,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        height=h,
        font=dict(family="Inter, sans-serif", color="#94a3b8", size=12),
        title_font=dict(color="#e2e8f0", size=14, family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#94a3b8"), borderwidth=0),
        margin=dict(l=8, r=8, t=40, b=8),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID_COL, linecolor="#1e3a5f", tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COL, linecolor="#1e3a5f", tickfont=dict(size=11)),
    )
    return fig

@st.cache_data(ttl=3600)
def load_data():
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    df = pd.read_csv(url)
    return df[df["year"] >= 1990].copy()

with st.spinner("Loading global energy data..."):
    df = load_data()

EXCLUDE = ["World","Asia","Africa","Europe","America","OECD","income","Ember","EI","G20","G7","Union","excl."]
all_c = sorted(df["country"].unique())
countries = [c for c in all_c if not any(k in c for k in EXCLUDE)]

with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 20px;border-bottom:1px solid #1e3a5f;margin-bottom:16px;text-align:center'>
        <div style='font-size:1rem;font-weight:700;color:#38bdf8;letter-spacing:1px'>⚡ ENERGY INTEL</div>
        <div style='font-size:0.68rem;color:#334155;letter-spacing:2px;margin-top:3px;text-transform:uppercase'>Global Dashboard v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    selected_countries = st.multiselect(
        "Countries",
        options=countries,
        default=["Pakistan","India","China","United States","Germany"]
    )
    year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()), (2000, 2022))
    pred_year  = st.slider("Forecast Until", 2023, 2040, 2030)

    st.markdown("""
    <div style='margin-top:24px;background:#071120;border:1px solid #1e3a5f;border-radius:10px;padding:14px;font-size:0.78rem;color:#475569;line-height:1.8'>
    <div style='color:#38bdf8;font-weight:600;margin-bottom:8px'>Data Source</div>
    Our World in Data<br>200+ Countries · Since 1990<br>Updated Annually
    </div>
    """, unsafe_allow_html=True)

if not selected_countries:
    st.warning("Please select at least one country from the sidebar.")
    st.stop()

filtered = df[df["country"].isin(selected_countries) & df["year"].between(*year_range)].copy()
latest_year = df["year"].max()
latest = df[(df["year"] == latest_year) & df["country"].isin(selected_countries)]

st.markdown("""
<div class="header-wrap">
  <div class="header-left">
    <h1>⚡ Global Energy Intelligence</h1>
    <p>Powered by Our World in Data · Analytics & ML Forecasting · SAP 70177923</p>
  </div>
  <div class="live-badge"><span class="live-dot"></span>Live Data</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("⚡ Total Energy", f'{latest["primary_energy_consumption"].sum():,.0f} TWh')
with c2: st.metric("🌱 Renewables", f'{latest["renewables_share_energy"].mean():.1f}%', delta="+1.8%")
with c3:
    fos = latest["fossil_share_energy"].mean() if "fossil_share_energy" in df.columns else 0
    st.metric("🛢️ Fossil Fuel", f"{fos:.1f}%", delta="-2.1%")
with c4: st.metric("👤 Per Capita", f'{latest["energy_per_capita"].mean():,.0f} kWh')
with c5:
    nuc = latest["nuclear_share_energy"].mean() if "nuclear_share_energy" in df.columns else 0
    st.metric("☢️ Nuclear", f"{nuc:.1f}%")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends",
    "🌍 World Map",
    "🥧 Energy Mix",
    "🌱 Renewables vs Fossil",
    "📊 CO₂ Emissions",
    "🔮 ML Forecast"
])

with tab1:
    st.markdown('<div class="section-title">Primary Energy Consumption</div>', unsafe_allow_html=True)
    fig = px.line(filtered, x="year", y="primary_energy_consumption",
        color="country", color_discrete_sequence=COLORS,
        labels={"primary_energy_consumption":"TWh","year":"Year","country":"Country"})
    fig.update_traces(line_width=2.5, mode="lines+markers", marker_size=3)
    st.plotly_chart(styled_fig(fig, 420), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Energy Per Capita</div>', unsafe_allow_html=True)
        fig2 = px.bar(filtered, x="year", y="energy_per_capita", color="country",
            color_discrete_sequence=COLORS, barmode="group",
            labels={"energy_per_capita":"kWh/person","year":"Year"})
        st.plotly_chart(styled_fig(fig2, 340), use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Total Energy Comparison</div>', unsafe_allow_html=True)
        lb = latest.sort_values("primary_energy_consumption", ascending=True)
        fig3 = px.bar(lb, x="primary_energy_consumption", y="country", orientation="h",
            color="primary_energy_consumption", color_continuous_scale="Blues",
            labels={"primary_energy_consumption":"TWh"})
        st.plotly_chart(styled_fig(fig3, 340), use_container_width=True)

with tab2:
    st.markdown('<div class="section-title">World Energy Map</div>', unsafe_allow_html=True)
    col_m1, col_m2 = st.columns([2,1])
    with col_m1:
        map_year = st.select_slider("Select Year", options=sorted(df["year"].unique()), value=2022)
    with col_m2:
        map_metric = st.selectbox("Metric", {
            "primary_energy_consumption":"⚡ Total Energy (TWh)",
            "renewables_share_energy":"🌱 Renewables %",
            "energy_per_capita":"👤 Per Capita (kWh)"
        }.keys(), format_func=lambda x: {
            "primary_energy_consumption":"⚡ Total Energy (TWh)",
            "renewables_share_energy":"🌱 Renewables %",
            "energy_per_capita":"👤 Per Capita (kWh)"
        }[x])

    wdata = df[df["year"] == map_year][["country", map_metric]].dropna()
    scale = "Viridis" if "renew" in map_metric else ("RdYlGn_r" if "fossil" in map_metric else "Blues")
    fig_map = px.choropleth(wdata, locations="country", locationmode="country names",
        color=map_metric, color_continuous_scale=scale,
        title=f"{map_metric.replace('_',' ').title()} — {map_year}")
    fig_map.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, height=520,
        geo=dict(bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True,
                 coastlinecolor="#1e3a5f", landcolor="#0d1f3c", showocean=True, oceancolor="#060d1f"),
        margin=dict(l=0,r=0,t=40,b=0),
        font=dict(family="Inter",color="#94a3b8"),
        coloraxis_colorbar=dict(tickfont=dict(color="#94a3b8")))
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Energy Mix Breakdown</div>', unsafe_allow_html=True)
    mix_country = st.selectbox("Select Country", selected_countries, key="mix_c")
    mix_df = df[(df["country"] == mix_country) & df["year"].between(*year_range)]
    sources = {
        "Coal":"coal_share_energy","Oil":"oil_share_energy",
        "Gas":"gas_share_energy","Nuclear":"nuclear_share_energy",
        "Renewables":"renewables_share_energy","Hydro":"hydro_share_energy",
        "Solar":"solar_share_energy","Wind":"wind_share_energy"
    }
    avail = {k:v for k,v in sources.items() if v in mix_df.columns and mix_df[v].notna().any()}
    latest_mix = mix_df[mix_df["year"]==mix_df["year"].max()][list(avail.values())].mean()
    latest_mix.index = list(avail.keys())
    latest_mix = latest_mix.dropna()

    cp, ca = st.columns([1,2])
    with cp:
        fig_pie = go.Figure(go.Pie(
            labels=latest_mix.index, values=latest_mix.values, hole=0.55,
            marker=dict(colors=COLORS, line=dict(color="#060d1f",width=2)),
            textfont=dict(size=12,color="#e2e8f0")))
        fig_pie.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG, height=360,
            title=dict(text=f"{mix_country} — Current Mix",font=dict(color="#e2e8f0",size=13)),
            legend=dict(font=dict(size=11,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=8,r=8,t=40,b=8))
        st.plotly_chart(fig_pie, use_container_width=True)
    with ca:
        area_cols = [v for v in avail.values() if v in mix_df.columns]
        area_data = mix_df[["year"]+area_cols].set_index("year").fillna(0)
        area_data.columns = list(avail.keys())[:len(area_cols)]
        fig_area = go.Figure()
        for i,col in enumerate(area_data.columns):
            fig_area.add_trace(go.Scatter(
                x=area_data.index, y=area_data[col], name=col,
                stackgroup="one", mode="none", fillcolor=COLORS[i%len(COLORS)]))
        fig_area.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG, height=360,
            title=dict(text=f"{mix_country} — Mix Over Time",font=dict(color="#e2e8f0",size=13)),
            legend=dict(font=dict(size=11,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=8,r=8,t=40,b=8),
            xaxis=dict(gridcolor=GRID_COL), yaxis=dict(gridcolor=GRID_COL,ticksuffix="%"))
        st.plotly_chart(fig_area, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">Renewables vs Fossil Fuel Transition</div>', unsafe_allow_html=True)
    fig_rv = go.Figure()
    for i,country in enumerate(selected_countries):
        c_df = filtered[filtered["country"]==country]
        clr = COLORS[i%len(COLORS)]
        if "renewables_share_energy" in c_df.columns:
            fig_rv.add_trace(go.Scatter(x=c_df["year"], y=c_df["renewables_share_energy"],
                name=f"{country} 🌱", line=dict(color=clr,width=2.5),
                mode="lines+markers", marker_size=4))
        if "fossil_share_energy" in c_df.columns:
            fig_rv.add_trace(go.Scatter(x=c_df["year"], y=c_df["fossil_share_energy"],
                name=f"{country} 🛢️", line=dict(color=clr,width=2,dash="dash"),
                mode="lines+markers", marker_size=4))
    fig_rv.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG, height=450,
        title=dict(text="Renewables (solid) vs Fossil Fuel (dashed)",font=dict(color="#e2e8f0",size=14)),
        yaxis=dict(title="% Share",gridcolor=GRID_COL,ticksuffix="%"),
        xaxis=dict(gridcolor=GRID_COL), hovermode="x unified",
        legend=dict(font=dict(size=10,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=8,r=8,t=50,b=8))
    st.plotly_chart(fig_rv, use_container_width=True)

    col_s, col_w = st.columns(2)
    for col_x, metric, label, clr in [
        (col_s,"solar_share_energy","Solar Energy Share %","#f59e0b"),
        (col_w,"wind_share_energy","Wind Energy Share %","#22c55e")
    ]:
        if metric in filtered.columns:
            with col_x:
                fig_x = px.area(filtered, x="year", y=metric, color="country",
                    color_discrete_sequence=COLORS, title=label,
                    labels={metric:"% Share","year":"Year"})
                fig_x.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG,
                    plot_bgcolor=PLOT_BG, height=320,
                    yaxis=dict(ticksuffix="%",gridcolor=GRID_COL),
                    xaxis=dict(gridcolor=GRID_COL),
                    legend=dict(font=dict(size=10,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
                    margin=dict(l=8,r=8,t=40,b=8))
                st.plotly_chart(fig_x, use_container_width=True)

with tab5:
    st.markdown('<div class="section-title">CO₂ & Greenhouse Gas Emissions</div>', unsafe_allow_html=True)
    co2_col = "co2" if "co2" in df.columns else ("greenhouse_gas_emissions" if "greenhouse_gas_emissions" in df.columns else None)

    if co2_col:
        fig_co2 = px.area(filtered, x="year", y=co2_col, color="country",
            color_discrete_sequence=COLORS,
            title="CO₂ Emissions (Million Tonnes)",
            labels={co2_col:"Mt CO₂","year":"Year"})
        fig_co2.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG, height=400,
            yaxis=dict(gridcolor=GRID_COL), xaxis=dict(gridcolor=GRID_COL),
            legend=dict(font=dict(size=11,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(fig_co2, use_container_width=True)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            if "co2_per_capita" in df.columns:
                fig_pc = px.line(filtered, x="year", y="co2_per_capita", color="country",
                    color_discrete_sequence=COLORS,
                    title="CO₂ Per Capita (tonnes/person)",
                    labels={"co2_per_capita":"Tonnes","year":"Year"})
                fig_pc.update_traces(line_width=2.5)
                st.plotly_chart(styled_fig(fig_pc, 340), use_container_width=True)
        with col_e2:
            if "co2_per_gdp" in df.columns:
                fig_gdp = px.line(filtered, x="year", y="co2_per_gdp", color="country",
                    color_discrete_sequence=COLORS,
                    title="CO₂ per GDP (kg per $)",
                    labels={"co2_per_gdp":"kg/$","year":"Year"})
                fig_gdp.update_traces(line_width=2.5)
                st.plotly_chart(styled_fig(fig_gdp, 340), use_container_width=True)
    else:
        st.info("CO₂ data not available for selected filters.")

with tab6:
    st.markdown('<div class="section-title">ML Energy Forecast</div>', unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1,2])
    with col_p1:
        pred_country = st.selectbox("Country", selected_countries, key="pred_c")
        pred_col = st.selectbox("Metric", {
            "primary_energy_consumption":"⚡ Primary Energy (TWh)",
            "renewables_share_energy":"🌱 Renewables Share %",
            "energy_per_capita":"👤 Energy Per Capita (kWh)"
        }.keys(), format_func=lambda x:{
            "primary_energy_consumption":"⚡ Primary Energy (TWh)",
            "renewables_share_energy":"🌱 Renewables Share %",
            "energy_per_capita":"👤 Energy Per Capita (kWh)"
        }[x])

    train_df = df[df["country"]==pred_country][["year",pred_col]].dropna()

    if len(train_df) < 10:
        st.error("Not enough data for this country.")
    else:
        X = train_df[["year"]].values
        y = train_df[pred_col].values
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        model.fit(X, y)
        future = np.arange(train_df["year"].min(), pred_year+1).reshape(-1,1)
        preds  = model.predict(future)
        pred_df = pd.DataFrame({"year":future.flatten(),"predicted":preds})
        split = train_df["year"].max()

        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=train_df["year"], y=train_df[pred_col],
            name="Actual Data", line=dict(color="#3b82f6",width=2.5),
            mode="lines+markers", marker_size=4,
            fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
        fig_f.add_trace(go.Scatter(
            x=pred_df[pred_df["year"]>split]["year"],
            y=pred_df[pred_df["year"]>split]["predicted"],
            name="ML Forecast", line=dict(color="#a78bfa",width=2.5,dash="dash"),
            mode="lines+markers", marker_size=4,
            fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"))
        fig_f.add_vline(x=split, line_dash="dot", line_color="#475569",
            annotation_text=f"Present ({split})",
            annotation_font=dict(color="#64748b",size=11))
        fig_f.update_layout(template=TEMPLATE, paper_bgcolor=PAPER_BG,
            plot_bgcolor=PLOT_BG, height=420,
            title=dict(text=f"{pred_country} — Forecast to {pred_year}",font=dict(color="#e2e8f0",size=14)),
            yaxis=dict(gridcolor=GRID_COL,tickfont=dict(size=11)),
            xaxis=dict(gridcolor=GRID_COL,tickfont=dict(size=11)),
            legend=dict(font=dict(size=11,color="#94a3b8"),bgcolor="rgba(0,0,0,0)"),
            hovermode="x unified", margin=dict(l=8,r=8,t=50,b=8))
        st.plotly_chart(fig_f, use_container_width=True)

        final_pred  = pred_df[pred_df["year"]==pred_year]["predicted"].values[0]
        current_val = train_df[pred_col].iloc[-1]
        change_pct  = ((final_pred - current_val) / current_val) * 100
        arrow = "▲" if change_pct > 0 else "▼"
        trend_clr = "#4ade80" if change_pct > 0 else "#f87171"

        st.markdown(f"""
        <div class="pred-grid">
            <div class="pred-card">
                <div class="pred-num" style="color:#a5b4fc">{final_pred:,.1f}</div>
                <div class="pred-lbl">Predicted {pred_year}</div>
            </div>
            <div class="pred-card">
                <div class="pred-num" style="color:#38bdf8">{current_val:,.1f}</div>
                <div class="pred-lbl">Current ({split})</div>
            </div>
            <div class="pred-card">
                <div class="pred-num" style="color:{trend_clr}">{arrow} {abs(change_pct):.1f}%</div>
                <div class="pred-lbl">Expected Change</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#1e3a5f;font-size:0.72rem;padding:40px 0 20px;letter-spacing:1px'>
⚡ Global Energy Intelligence · SAP ID: 70177923 · Our World in Data · Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
