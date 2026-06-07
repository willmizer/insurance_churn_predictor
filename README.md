# Churn Analytics Dashboard

[![Live Demo](https://img.shields.io/badge/AWS-Live_Demo-FF9900?style=for-the-badge&logo=amazon-aws)](https://insurance-churn-predictor.duckdns.org/) - currently down

A professional-grade web application built with **Flask** and **XGBoost** to predict student/customer churn. This project utilizes an ensemble machine learning approach to provide real-time retention insights and executive-level data visualization.

## Key Features
* **Dual-Model Architecture:** Uses a multi-model ensemble for active users and an optimized XGBoost model for retired demographics.
* **Real-Time Analytics:** Integration with **SHAP** for explainable AI, identifying the top drivers for churn and retention.
* **Executive Dashboard:** Interactive UI for high-level churn metrics, value-at-risk assessments, and sales recommendations.
* **Production Ready:** Configured for deployment on AWS using Nginx and Gunicorn.

## Tech Stack
* **Backend:** Python (Flask)
* **Machine Learning:** XGBoost, Scikit-Learn, Joblib
* **Explainable AI:** SHAP (Shapley Additive Explanations)
* **Frontend:** HTML5, CSS3, Chart.js
* **Infrastructure:** AWS (EC2), Nginx, Gunicorn

## Project Structure
* `/Web`: Core Flask application and UI templates.
* `/pipeline`: Data processing scripts and cleaned datasets.
* `/Web/models`: Trained serialization files (.pkl) and thresholds.

## Usage & Copyright
© 2026 Will Mizer. All rights reserved.

## Repository Status: View-Only
This repository is for **code review and portfolio demonstration purposes only**. 

* **No License:** All rights reserved. No permission is granted for downloading, running, or modifying this repository.
* **Missing Dependencies:** Critical machine learning models and datasets have been omitted to protect logic and data privacy.
