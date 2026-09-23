import os
import numpy as np
import pandas as pd
import joblib
import shap
import plotly.express as px
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "pipeline")
VAR_DESC_PATH = os.path.join(BASE_DIR, "variable_descriptions.txt")

DATASETS = {
    "Active customers": {"key": "active", "file": "active_clean.csv", "model": "ensemble_model.pkl"},
    "Retired customers": {"key": "retired", "file": "retired_clean.csv", "model": "xgb_retired_optimized.pkl"},
}

RECOMMENDATION_RULES = {
    'CLAIM3YEARS': 'Review recent claim history and offer claims assistance.',
    'P1_EMP_STATUS': 'Update employment details and check for eligible discounts.',
    'BUS_USE': 'Verify commercial usage terms and offer business policy add-ons.',
    'AD_BUILDINGS': 'Review building accidental damage coverage limits.',
    'RISK_RATED_AREA_B': 'Assess regional risk factors and suggest mitigation measures.',
    'MTA_FAP': 'Review mid-term adjustments and premium impacts.',
    'LAST_ANN_PREM_GROSS': 'Offer loyalty discount or flexible payment plan.',
    'P1_AGE': 'Check age-related policy benefits (e.g., senior discounts).',
    'PROP_AGE': 'Suggest property maintenance review or updated valuation.',
    'DAYS_TO_BIND': 'Streamline binding process for future renewals.',
    'PAYMENT_PureDD': 'Suggest switching to automated payments for convenience.',
    'NCD_GRANTED_YEARS_B': 'Highlight No Claims Discount benefits.',
    'LEGAL_ADDON_POST_REN': 'Highlight value of legal protection in case of disputes.',
    'LEGAL_ADDON_PRE_REN': 'Highlight value of legal protection in case of disputes.',
    'MAX_DAYS_UNOCC': 'Discuss occupancy terms and range of unoccupancy cover.',
    'SUM_INSURED_CONTENTS': 'Review coverage limits to ensure adequate protection.',
    'RISK_RATED_AREA_C': 'Explain regional risk factors affecting premium.',
}

FEATURE_ORDER = [
    'CLAIM3YEARS', 'BUS_USE', 'CLERICAL', 'AD_BUILDINGS', 'RISK_RATED_AREA_B',
    'SUM_INSURED_BUILDINGS', 'NCD_GRANTED_YEARS_B', 'AD_CONTENTS', 'RISK_RATED_AREA_C',
    'SUM_INSURED_CONTENTS', 'NCD_GRANTED_YEARS_C', 'CONTENTS_COVER', 'BUILDINGS_COVER',
    'SPEC_SUM_INSURED', 'SPEC_ITEM_PREM', 'UNSPEC_HRP_PREM', 'P1_POLICY_REFUSED',
    'P1_SEX', 'APPR_ALARM', 'APPR_LOCKS', 'BEDROOMS', 'ROOF_CONSTRUCTION',
    'WALL_CONSTRUCTION', 'FLOODING', 'LISTED', 'MAX_DAYS_UNOCC', 'NEIGH_WATCH',
    'OWNERSHIP_TYPE', 'PAYING_GUESTS', 'PROP_TYPE', 'SAFE_INSTALLED', 'SEC_DISC_REQ',
    'SUBSIDENCE', 'YEARBUILT', 'PAYMENT_FREQUENCY',
    'LEGAL_ADDON_PRE_REN', 'LEGAL_ADDON_POST_REN', 'HOME_EM_ADDON_PRE_REN',
    'HOME_EM_ADDON_POST_REN', 'GARDEN_ADDON_PRE_REN', 'GARDEN_ADDON_POST_REN',
    'KEYCARE_ADDON_PRE_REN', 'KEYCARE_ADDON_POST_REN', 'HP1_ADDON_PRE_REN',
    'HP1_ADDON_POST_REN', 'HP2_ADDON_PRE_REN', 'HP2_ADDON_POST_REN',
    'HP3_ADDON_PRE_REN', 'HP3_ADDON_POST_REN', 'MTA_FLAG', 'MTA_FAP', 'MTA_APRP',
    'LAST_ANN_PREM_GROSS', 'POL_STATUS', 'HAS_MTA', 'HAS_ADJUSTED', 'P1_AGE', 'PROP_AGE', 'DAYS_TO_BIND',
    'IS_STANDARD_OCCUPANCY', 'HAS_PT_EMP', 'MAR_STATUS_Couple', 'MAR_STATUS_Solo',
    'DAYS_SINCE_MTA', 'QUOTE_MONTH', 'QUOTE_WEEKDAY',
    'PAYMENT_METHOD',
    'EMP_STATUS_Carer', 'EMP_STATUS_Disabled', 'EMP_STATUS_Employed',
    'EMP_STATUS_Houseperson', 'EMP_STATUS_None', 'EMP_STATUS_Other',
    'EMP_STATUS_Student', 'EMP_STATUS_Unemployed', 'EMP_STATUS_Voluntary'
]


@st.cache_resource(show_spinner=False)
def load_feature_mapping():
    mapping = {}
    try:
        with open(VAR_DESC_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, val = line.split(":", 1)
                mapping[key.strip()] = val.strip()
    except FileNotFoundError:
        pass
    return mapping


@st.cache_resource(show_spinner=False)
def load_model(model_file):
    return joblib.load(os.path.join(MODELS_DIR, model_file))


@st.cache_resource(show_spinner=False)
def load_active_threshold():
    try:
        with open(os.path.join(MODELS_DIR, "ensemble_threshold.txt")) as f:
            return float(f.read().strip())
    except Exception:
        return 0.5


def shap_to_delta_percent(shap_value, base_value):
    base_prob = 1.0 / (1.0 + np.exp(-float(base_value)))
    feature_prob = 1.0 / (1.0 + np.exp(-(float(base_value) + float(shap_value))))
    return round((feature_prob - base_prob) * 100, 2)


@st.cache_data(show_spinner=True)
def process_dataset(dataset_label):
    info = DATASETS[dataset_label]
    model = load_model(info["model"])
    df = pd.read_csv(os.path.join(DATA_DIR, info["file"]))
    df.columns = df.columns.astype(str)

    if hasattr(model, "feature_names_in_"):
        feature_names = [str(x) for x in model.feature_names_in_]
    elif hasattr(model, "get_booster"):
        feature_names = [str(x) for x in model.get_booster().feature_names]
    else:
        feature_names = [c for c in FEATURE_ORDER if c in df.columns]

    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    X = df[feature_names]
    X.columns = X.columns.astype(str)

    probs = model.predict_proba(X)[:, 1]
    df["Churn Probability Raw"] = probs * 100

    thresh = load_active_threshold() if info["key"] == "active" else 0.5

    def classify(p):
        prob = p / 100.0
        if prob < thresh:
            return "Retain"
        elif prob < 0.85:
            return "Likely Churn"
        return "Certain Churn"

    df["Status"] = df["Churn Probability Raw"].apply(classify)
    df["Churn Probability"] = df["Churn Probability Raw"].apply(lambda p: f"{p:.2f}%")

    # SHAP explainability (fast enough on the full dataset for this data size)
    explainer_model = model
    if hasattr(model, "named_estimators_"):
        explainer_model = model.named_estimators_.get("xgb", list(model.named_estimators_.values())[0])
    explainer = shap.TreeExplainer(explainer_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else np.atleast_1d(base_value)[0]

    feats = X.columns.tolist()
    top_drivers = []
    recommendations = []
    shap_full_rows = []
    for i in range(len(X)):
        contributions = dict(zip(feats, shap_values[i]))
        shap_full_rows.append(contributions)
        sorted_feats = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        top_list = [(f, shap_to_delta_percent(v, base_value)) for f, v in sorted_feats]
        top_drivers.append(top_list)

        top_positive = next((f for f, v in top_list if v > 0), None)
        recommendations.append(RECOMMENDATION_RULES.get(top_positive, "Standard Retention Check"))

    df["Top Driver"] = [d[0][0] if d else None for d in top_drivers]
    df["Recommendation"] = recommendations

    # aggregate driver impact across the whole dataset
    feature_sums = {f: 0.0 for f in feats}
    for row in shap_full_rows:
        for f, v in row.items():
            feature_sums[f] += v
    n = len(shap_full_rows)
    avg_impact = {k: v / n for k, v in feature_sums.items()}

    mapping = load_feature_mapping()
    sorted_up = sorted(avg_impact.items(), key=lambda kv: kv[1], reverse=True)
    top_churn = [{"feature": mapping.get(f, f), "impact": round(v, 3)} for f, v in sorted_up if v > 0][:5]
    sorted_down = sorted(avg_impact.items(), key=lambda kv: kv[1])
    top_retention = [{"feature": mapping.get(f, f), "impact": round(abs(v), 3)} for f, v in sorted_down if v < 0][:5]

    total = len(df)
    certain_churn = int((df["Status"] == "Certain Churn").sum())
    churn_rate = round(certain_churn / total * 100, 1) if total else 0.0

    risk_likely = float(df.loc[df["Status"] == "Likely Churn", "LAST_ANN_PREM_GROSS"].sum()) if "LAST_ANN_PREM_GROSS" in df.columns else 0.0
    risk_certain = float(df.loc[df["Status"] == "Certain Churn", "LAST_ANN_PREM_GROSS"].sum()) if "LAST_ANN_PREM_GROSS" in df.columns else 0.0
    value_at_risk = risk_likely + risk_certain
    revenue_saved = value_at_risk * 0.20

    summary = {
        "total_customers": total,
        "churn_rate": churn_rate,
        "value_at_risk": value_at_risk,
        "revenue_saved": revenue_saved,
    }

    display_cols = [c for c in df.columns if c not in ("Churn Probability Raw",)]
    return df[display_cols], summary, top_churn, top_retention


st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide")

# Keep the modebar so Streamlit's own fullscreen-expand button (injected into it)
# still shows, but strip every other Plotly tool (zoom, pan, select, download, etc).
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "zoom2d", "pan2d", "select2d", "lasso2d",
        "zoomIn2d", "zoomOut2d", "autoScale2d", "resetScale2d",
        "toImage", "hoverClosestCartesian", "hoverCompareCartesian",
        "toggleSpikelines",
    ],
}
st.markdown(
    """
    <style>
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Churn Analytics Dashboard")
st.caption(
    "A dual-model ensemble (XGBoost + Random Forest + LightGBM for active policyholders, "
    "a tuned XGBoost model for retired policyholders) predicts home-insurance customer churn, "
    "with SHAP explainability identifying the top drivers behind every prediction."
)

dataset_label = st.radio("Customer segment", list(DATASETS.keys()), horizontal=True)

with st.spinner("Scoring customers and computing SHAP explanations..."):
    df, summary, top_churn, top_retention = process_dataset(dataset_label)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total customers", f"{summary['total_customers']:,}")
c2.metric("Certain-churn rate", f"{summary['churn_rate']}%")
c3.metric("Value at risk", f"${summary['value_at_risk']:,.0f}")
c4.metric("Est. revenue saved (20% retained)", f"${summary['revenue_saved']:,.0f}")

st.divider()

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("Top churn drivers")
    if top_churn:
        fig = px.bar(
            pd.DataFrame(top_churn), x="impact", y="feature", orientation="h",
            labels={"impact": "Avg. probability impact (pts)", "feature": ""},
            color_discrete_sequence=["#d62728"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    else:
        st.info("No positive churn drivers found in this segment.")

with col_right:
    st.subheader("Top retention drivers")
    if top_retention:
        fig = px.bar(
            pd.DataFrame(top_retention), x="impact", y="feature", orientation="h",
            labels={"impact": "Avg. probability impact (pts)", "feature": ""},
            color_discrete_sequence=["#2ca02c"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
    else:
        st.info("No retention drivers found in this segment.")

st.divider()
st.subheader("Customer-level predictions")

status_options = ["All"] + sorted(df["Status"].unique().tolist())
fc1, fc2, fc3 = st.columns([1, 1, 2])
status_filter = fc1.selectbox("Filter by status", status_options)
sort_order = fc2.selectbox("Sort by churn probability", ["Highest first", "Lowest first"])
row_limit = fc3.slider("Rows to display", min_value=10, max_value=500, value=50, step=10)

table_df = df.copy()
if status_filter != "All":
    table_df = table_df[table_df["Status"] == status_filter]
table_df = table_df.sort_values(
    by="Churn Probability", ascending=(sort_order == "Lowest first"),
    key=lambda s: s.str.rstrip("%").astype(float),
)

show_cols = [c for c in ["Status", "Churn Probability", "Top Driver", "Recommendation"] if c in table_df.columns]
st.dataframe(table_df[show_cols].head(row_limit), width="stretch", height=420)

with st.expander("Feature glossary"):
    mapping = load_feature_mapping()
    st.dataframe(pd.DataFrame(sorted(mapping.items()), columns=["Feature", "Description"]), width="stretch", height=300)
