# Churn Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/Streamlit-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](TODO_STREAMLIT_URL)

An interactive Streamlit dashboard built with **XGBoost**, **Random Forest**, and **LightGBM** to predict home-insurance customer churn. This project uses an ensemble machine learning approach to provide real-time retention insights and executive-level data visualization.

## Key Features
* **Dual-Model Architecture:** A multi-model ensemble (XGBoost + Random Forest + LightGBM) for active customers, and an optimized XGBoost model for retired customers.
* **Explainable AI:** SHAP (TreeExplainer) identifies the top drivers behind every prediction, computed live on the full dataset.
* **Executive Dashboard:** Interactive summary metrics, value-at-risk assessment, and top churn/retention driver charts.
* **Customer-level table:** Filter by churn status, sort by churn probability, and see the top SHAP-driven recommendation per customer.

## Tech Stack
* **App:** Python (Streamlit)
* **Machine Learning:** XGBoost, Random Forest, LightGBM, Scikit-Learn, Joblib
* **Explainable AI:** SHAP (Shapley Additive Explanations)
* **Charts:** Plotly

## Project Structure
* `app.py` — Streamlit app (entry point).
* `models/` — Trained model files (`ensemble_model.pkl`, `xgb_retired_optimized.pkl`) and the active-model decision threshold.
* `pipeline/` — Cleaned datasets (`active_clean.csv`, `retired_clean.csv`) and the training script (`train_ensemble.py`).
* `variable_descriptions.txt` — Human-readable descriptions for every model feature (shown in the app's feature glossary).

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Usage & Copyright
© 2026 Will Mizer.
