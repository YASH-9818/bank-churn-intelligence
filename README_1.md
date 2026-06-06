# 🏦 Bank Customer Churn Intelligence Platform

> Predictive Modeling and Risk Scoring for European Central Bank Customer Churn Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=flat-square&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple?style=flat-square&logo=plotly)

---

## 📌 Overview

This project builds a full-stack churn intelligence system for a European retail bank using machine learning. It identifies customers likely to leave, assigns them a churn probability score, and provides actionable retention insights through an interactive dashboard.

**Dataset:** 10,000 customers across France, Germany, and Spain  
**Target variable:** `Exited` (1 = churned, 0 = retained)  
**Overall churn rate:** 20.4%

---

## 🎯 Key Findings

| Segment | Churn Rate | Insight |
|---------|-----------|---------|
| Germany customers | 32.4% | 2× higher than France/Spain |
| Age 51–60 | 56.2% | Highest risk age group |
| 4 products | 100% | All customers churned |
| 2 products | 7.6% | Optimal engagement tier |
| Inactive members | 26.9% | 1.9× higher than active |
| Female customers | 25.1% | vs 16.5% for males |

---

## 🤖 Models Trained

| Model | ROC-AUC |
|-------|---------|
| Gradient Boosting | ~0.87 |
| Random Forest | ~0.86 |
| Logistic Regression | ~0.77 |
| Decision Tree | ~0.73 |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bank-churn-intelligence.git
cd bank-churn-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (run once, ~2 minutes)
```bash
python train_models.py
```

### 4. Launch the dashboard
```bash
streamlit run app.py
# or on Windows:
python -m streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI metrics, churn by geography, products, gender, activity |
| **Deep Dive** | Age/balance distributions, correlation heatmap, tenure trends |
| **Model Performance** | Accuracy, Precision, Recall, F1, ROC-AUC + confusion matrices |
| **Feature Importance** | Random Forest feature weights + engineered features |
| **Risk Calculator** | Enter customer details → live churn probability score |
| **Scenario Simulator** | Model retention campaign impact on portfolio churn rate |

---

## 📁 Project Structure

```
bank-churn-intelligence/
├── app.py                  # Streamlit dashboard
├── train_models.py         # ML pipeline
├── requirements.txt        # Dependencies
├── README.md               # This file
├── European_Bank.csv       # Dataset (10,000 customers)
└── models/                 # Auto-generated after training
    ├── gradient_boosting.pkl
    ├── random_forest.pkl
    ├── decision_tree.pkl
    ├── logistic_regression.pkl
    ├── metrics.json
    ├── feature_importance.json
    ├── feature_cols.json
    └── test_predictions.csv
```

---

## ⚙️ Feature Engineering

Beyond the raw dataset columns, the following features were derived:

- **BalanceToSalary** — financial stress indicator
- **ProductDensity** — products per year of tenure
- **EngagementProduct** — activity × product interaction
- **AgeTenureInteraction** — combined demographic depth
- **HighBalance** — binary flag for balance > €100K
- **SeniorCustomer** — binary flag for age ≥ 45

---

## 📋 Requirements

- Python 3.9 or higher
- All files in the same directory
- Run `train_models.py` before launching the app

---

## 🏛️ Project Context

Built as part of the **Unified Mentor × European Central Bank** internship project on Predictive Modeling and Risk Scoring for Bank Customer Churn.
