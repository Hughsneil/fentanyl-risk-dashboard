import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go


# Page Config
st.set_page_config(
    page_title="Fentanyl Risk Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #07111f 0%, #0b1324 100%);
        color: white;
    }

    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 100%;
    }

    h1, h2, h3 {
        color: white !important;
    }

    .hero-card {
        background: linear-gradient(135deg, rgba(35,53,84,0.95), rgba(18,28,44,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 18px 22px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 0.8rem;
    }

    .toolbar-card, .panel-card, .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    }

    .metric-card {
        padding: 12px 16px;
        min-height: 88px;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #9fb3c8;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: white;
    }

    .metric-sub {
        font-size: 0.75rem;
        color: #6fb3ff;
        margin-top: 4px;
    }

    .small-note {
        font-size: 0.82rem;
        color: #9fb3c8;
    }

    div[data-testid="stTabs"] button {
        border-radius: 10px !important;
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    .stSelectbox, .stNumberInput, .stMultiSelect {
        margin-bottom: -0.35rem;
    }

    .stButton>button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #2d7ff9, #00b8d9);
        color: white;
        font-weight: 600;
        height: 42px;
        width: 100%;
    }

    .footer-note {
        font-size: 0.78rem;
        color: #90a4b8;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Deployment_Drug_Related_Deaths_2012-2024.csv")

    # Date / Year
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
    else:
        df["Year"] = np.nan

    # Target
    df["Fentanyl_Related"] = (
        df["Fentanyl"].fillna("").astype(str).str.upper().eq("Y").astype(int)
    )

    # Binary drug features
    drug_cols = [
        "Heroin", "Cocaine", "Ethanol", "Benzodiazepine",
        "Methadone", "Meth/Amphetamine", "Xylazine",
        "Oxycodone", "Any Opioid"
    ]
    for col in drug_cols:
        if col in df.columns:
            df[col + "_bin"] = (
                df[col].fillna("").astype(str).str.upper().eq("Y").astype(int)
            )

    # Age group
    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 24, 34, 44, 54, 64, 120],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
        )

    return df

df = load_data()


# Load Trained Model
@st.cache_resource
def load_trained_model(data):
    model_path = "fentanyl_xgboost_model.pkl"

    feature_cols = [
        "Age", "Sex", "Race", "Ethnicity",
        "Residence County", "Injury County", "Death County",
        "Heroin_bin", "Cocaine_bin", "Ethanol_bin",
        "Benzodiazepine_bin", "Methadone_bin",
        "Meth/Amphetamine_bin", "Xylazine_bin",
        "Oxycodone_bin", "Any Opioid_bin"
    ]
    feature_cols = [c for c in feature_cols if c in data.columns]

    try:
        saved_object = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file not found: {model_path}. Please place the trained model file in the same folder as app.py.")
        st.stop()

    if isinstance(saved_object, dict):
        model = saved_object.get("model")
        feature_cols = saved_object.get("feature_cols", feature_cols)
        model_auc = saved_object.get("auc", None)
    else:
        model = saved_object
        model_auc = None

    return model, feature_cols, model_auc

model, feature_cols, model_auc = load_trained_model(df)


# Header
st.markdown("""
<div class="hero-card">
    <h1 style="margin-bottom:6px;">Fentanyl-Related Accidental Death Risk Dashboard</h1>
</div>
""", unsafe_allow_html=True)


# Filter Toolbar
st.markdown('<div class="toolbar-card">', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns([1.25, 1, 1, 1])

years = sorted(df["Year"].dropna().astype(int).unique()) if "Year" in df.columns else [2012, 2024]

with f1:
    selected_years = st.select_slider(
        "Year Range",
        options=years,
        value=(min(years), max(years))
    )

with f2:
    sex_options = ["All"] + sorted(df["Sex"].dropna().astype(str).unique().tolist())
    selected_sex = st.selectbox("Sex", sex_options)

with f3:
    race_options = ["All"] + sorted(df["Race"].dropna().astype(str).unique().tolist())
    selected_race = st.selectbox("Race", race_options)

with f4:
    county_options = ["All"] + sorted(df["Death County"].dropna().astype(str).unique().tolist())
    selected_county = st.selectbox("Death County", county_options)

st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
filtered_df = df.copy()

if "Year" in filtered_df.columns:
    filtered_df = filtered_df[
        (filtered_df["Year"] >= selected_years[0]) &
        (filtered_df["Year"] <= selected_years[1])
    ]

if selected_sex != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_sex]

if selected_race != "All":
    filtered_df = filtered_df[filtered_df["Race"] == selected_race]

if selected_county != "All":
    filtered_df = filtered_df[filtered_df["Death County"] == selected_county]


# KPI Cards
total_records = len(filtered_df)
fentanyl_cases = int(filtered_df["Fentanyl_Related"].sum()) if total_records > 0 else 0
non_fentanyl_cases = total_records - fentanyl_cases
fentanyl_rate = fentanyl_cases / total_records if total_records > 0 else 0

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{total_records:,}</div>
        <div class="metric-sub">Filtered dataset size</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fentanyl Cases</div>
        <div class="metric-value">{fentanyl_cases:,}</div>
        <div class="metric-sub">Positive class count</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Fentanyl Rate</div>
        <div class="metric-value">{fentanyl_rate:.1%}</div>
        <div class="metric-sub">Share of filtered records</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    auc_display = f"{model_auc:.3f}" if model_auc is not None else "Loaded"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model ROC-AUC</div>
        <div class="metric-value">{auc_display}</div>
        <div class="metric-sub">Binary classification performance</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

# Main Layout
left, right = st.columns([1.9, 1])

# Left Panel: Tabs for Visualization
with left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Analytical Insights")

    tab1, tab2, tab3 = st.tabs([
        "Trend",
        "Demographics",
        "Risk Features",
    ])

    with tab1:
        trend_df = (
            filtered_df.groupby("Year")["Fentanyl_Related"]
            .sum()
            .reset_index()
            .rename(columns={"Fentanyl_Related": "Fentanyl Deaths"})
        )

        fig1 = px.line(
            trend_df,
            x="Year",
            y="Fentanyl Deaths",
            markers=True,
            template="plotly_dark"
        )
        fig1.update_traces(line=dict(width=3, color="#62b5ff"))
        fig1.update_layout(
            title="Yearly Trend of Fentanyl-Related Deaths",
            height=380,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        dem_option = st.selectbox(
            "Demographic Variable",
            ["Age_Group", "Sex", "Race"],
            key="demographic_option"
        )

        dem_df = (
            filtered_df.groupby(dem_option)["Fentanyl_Related"]
            .sum()
            .reset_index()
            .rename(columns={"Fentanyl_Related": "Fentanyl Deaths"})
        )

        fig2 = px.bar(
            dem_df,
            x=dem_option,
            y="Fentanyl Deaths",
            template="plotly_dark",
            color_discrete_sequence=["#62b5ff"]
        )
        fig2.update_layout(
            title=f"Fentanyl-Related Deaths by {dem_option}",
            height=380,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        drug_cols = [
            "Heroin_bin", "Cocaine_bin", "Ethanol_bin", "Benzodiazepine_bin",
            "Methadone_bin", "Meth/Amphetamine_bin", "Xylazine_bin",
            "Oxycodone_bin", "Any Opioid_bin"
        ]
        drug_cols = [c for c in drug_cols if c in filtered_df.columns]

        rows = []
        for col in drug_cols:
            temp = filtered_df[filtered_df[col] == 1]
            if len(temp) > 0:
                rows.append({
                    "Feature": col.replace("_bin", ""),
                    "Fentanyl Rate": temp["Fentanyl_Related"].mean()
                })

        risk_df = pd.DataFrame(rows).sort_values("Fentanyl Rate", ascending=True)

        fig3 = px.bar(
            risk_df,
            x="Fentanyl Rate",
            y="Feature",
            orientation="h",
            text=risk_df["Fentanyl Rate"].apply(lambda x: f"{x:.1%}"),
            template="plotly_dark",
            color_discrete_sequence=["#00d4ff"]
        )
        fig3.update_layout(
            title="Fentanyl Rate by Drug Involvement",
            height=380,
            margin=dict(l=10, r=10, t=45, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# Right Panel: Prediction
with right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.subheader("Prediction Panel")
    st.markdown("<div class='small-note'>Input demographic and substance-related features to estimate fentanyl-related risk.</div>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=35)
            sex_input = st.selectbox(
                "Sex",
                sorted(df["Sex"].dropna().astype(str).unique().tolist())
            )
            race_input = st.selectbox(
                "Race",
                sorted(df["Race"].dropna().astype(str).unique().tolist())
            )

        with c2:
            county_input = st.selectbox(
                "County",
                sorted(df["Death County"].dropna().astype(str).unique().tolist())
            )

            selected_drugs = st.multiselect(
                "Other Drug Involvement",
                [
                    "Heroin", "Cocaine", "Ethanol", "Benzodiazepine",
                    "Methadone", "Meth/Amphetamine", "Xylazine",
                    "Oxycodone", "Any Opioid"
                ],
                default=[]
            )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        input_df = pd.DataFrame([{
            "Age": age,
            "Sex": sex_input,
            "Race": race_input,
            "Ethnicity": "Unknown",
            "Residence County": county_input,
            "Injury County": county_input,
            "Death County": county_input,
            "Heroin_bin": 1 if "Heroin" in selected_drugs else 0,
            "Cocaine_bin": 1 if "Cocaine" in selected_drugs else 0,
            "Ethanol_bin": 1 if "Ethanol" in selected_drugs else 0,
            "Benzodiazepine_bin": 1 if "Benzodiazepine" in selected_drugs else 0,
            "Methadone_bin": 1 if "Methadone" in selected_drugs else 0,
            "Meth/Amphetamine_bin": 1 if "Meth/Amphetamine" in selected_drugs else 0,
            "Xylazine_bin": 1 if "Xylazine" in selected_drugs else 0,
            "Oxycodone_bin": 1 if "Oxycodone" in selected_drugs else 0,
            "Any Opioid_bin": 1 if "Any Opioid" in selected_drugs else 0
        }])

        input_df = input_df[[c for c in feature_cols if c in input_df.columns]]

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        if prob >= 0.7:
            risk_label = "High Risk"
            risk_color = "#ff5a6f"
        elif prob >= 0.4:
            risk_label = "Medium Risk"
            risk_color = "#f6c343"
        else:
            risk_label = "Low Risk"
            risk_color = "#2dd4bf"

        pred_text = "Fentanyl-Related" if pred == 1 else "Non-Fentanyl-Related"

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted Class</div>
                <div class="metric-value" style="font-size:1.15rem;">{pred_text}</div>
                <div class="metric-sub">Binary prediction result</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value" style="color:{risk_color};">{risk_label}</div>
                <div class="metric-sub">Based on predicted probability</div>
            </div>
            """, unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 28, "color": "white"}},
            title={"text": "Predicted Probability", "font": {"size": 15, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#62b5ff"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#12354f"},
                    {"range": [40, 70], "color": "#1b4d73"},
                    {"range": [70, 100], "color": "#255f92"}
                ]
            }
        ))
        gauge.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"}
        )
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("<div class='small-note'>Note: This tool is for analytical demonstration only and should not replace expert or public health judgment.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)