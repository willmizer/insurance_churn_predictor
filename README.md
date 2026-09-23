# Churn Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://insurance-churn-predictions.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)]()

**An interactive churn-prediction and explainability dashboard for home insurance policyholders.**

Built with an ensemble of XGBoost, Random Forest, and LightGBM, this project predicts which customers are likely to churn, explains *why* with SHAP, and estimates the dollar value at risk — the kind of tool a retention team would actually use to prioritize outreach.

---

## Overview

- **Dual-model architecture:** a multi-model ensemble for active customers, and a separately tuned XGBoost model for retired customers — two populations with different churn drivers.
- **Explainable AI:** SHAP (`TreeExplainer`) identifies the top drivers behind every individual prediction, computed live on the full dataset.
- **Executive dashboard:** summary metrics, value-at-risk assessment, and top churn/retention driver charts.
- **Customer-level table:** filter by churn status, sort by churn probability, and see the top SHAP-driven recommendation per customer.

## Tech Stack

- **App:** Python, Streamlit, Plotly
- **Machine Learning:** XGBoost, Random Forest, LightGBM (`VotingClassifier` ensemble), scikit-learn, Joblib
- **Explainable AI:** SHAP (Shapley Additive Explanations)

## Data Pipeline

### 1. Data
A home-insurance policy dataset (~70 engineered features covering coverage, claims history, property, and demographic attributes) is split into **active** and **retired** policyholder segments, since their churn dynamics differ enough to warrant separate models.

### 2. Modeling (`pipeline/train_ensemble.py`)
- **Active segment:** `GridSearchCV` over an XGBoost classifier (`max_depth`, `learning_rate`, `n_estimators`, `scale_pos_weight`), combined into a `VotingClassifier` ensemble with Random Forest and LightGBM.
- **Retired segment:** a separately grid-searched, optimized XGBoost model.
- Both models are deliberately tuned toward **recall on the churn class** over raw accuracy — in a retention context, missing a real churn risk is costlier than a false alarm that just triggers an unnecessary check-in.

### 3. Explainability & Dashboard (`app.py`)
- SHAP `TreeExplainer` runs on every row (not a sample — benchmarked at under 1.1 seconds even for the 146k-row retired segment) to get per-customer feature contributions.
- The strongest positive SHAP driver per customer maps to a canned recommendation (e.g. "Review recent claim history," "Highlight No Claims Discount benefits") via a rule table.
- Aggregate churn/retention drivers are the mean SHAP impact per feature across all customers.
- Value-at-risk sums `LAST_ANN_PREM_GROSS` for customers classified Likely/Certain Churn; estimated revenue saved assumes retaining 20% of that at-risk premium.

## Key Results

| Segment | Rows | Accuracy | Churn-class Recall | Churn-class Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Active** (ensemble) | 42,400 | 0.60–0.62 | **0.77–0.80** | 0.44–0.45 |
| **Retired** (XGBoost) | 146,622 | 0.67 | **0.71** | 0.46 |

The models trade precision for recall by design — catching ~3 out of 4 real churners is worth more to a retention program than a higher headline accuracy that misses them.

## Project Structure

```
insurance_project/
├── app.py                          # Streamlit app (entry point)
├── models/
│   ├── ensemble_model.pkl          # Active-segment ensemble
│   ├── xgb_retired_optimized.pkl   # Retired-segment XGBoost
│   └── ensemble_threshold.txt      # Active-model decision threshold
├── pipeline/
│   ├── train_ensemble.py           # Model training script
│   ├── active_clean.csv
│   └── retired_clean.csv
├── variable_descriptions.txt       # Feature glossary (shown in-app)
└── requirements.txt
```

## Run Locally

```bash
git clone https://github.com/willmizer/insurance_churn_predictor.git
cd insurance_churn_predictor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Recalibrate the precision/recall tradeoff with a cost-sensitive threshold rather than a fixed one.
- Add a "what-if" simulator (similar to the College Happiness project) to test the effect of a retention offer on churn probability.
- Track prediction drift over time as new policy data comes in.

## License

This project is shared for portfolio and educational purposes — feel free to explore the code. Please reach out before reusing it commercially.

© 2026 Will Mizer
