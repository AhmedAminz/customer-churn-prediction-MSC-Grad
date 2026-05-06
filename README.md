<div align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LightGBM-Tuned-brightgreen?style=for-the-badge&logo=leaflet&logoColor=white"/>
<img src="https://img.shields.io/badge/Accuracy-93.64%25-success?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit&logoColor=white"/>

# Customer Churn Prediction

> Predicting customer churn using gradient boosting on 500k+ records — MSC-KFS Data Science Committee Final Project

[![Live Demo](https://img.shields.io/badge/LIVE%20DEMO-%20Launch%20App%20%E2%86%92-FF4B4B?style=for-the-badge&logo=streamlit)](https://customer-churn-prediction-msc-grad-2hchzdkrpcugri87ksrcc5.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-AhmedAminz-181717?style=for-the-badge&logo=github)](https://github.com/AhmedAminz/customer-churn-prediction)

</div>

---

## Overview

A machine learning pipeline that predicts whether a customer will churn based on behavioral, demographic, and subscription data. Trained on 505,206 records with strict 5-fold cross-validation to ensure no data leakage.

---

## Results

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 0.840 |
| Gradient Boosting | 0.913 |
| XGBoost | 0.933 |
| Random Forest | 0.934 |
| LightGBM (default) | 0.9359 |
| **LightGBM (tuned)** | **0.9364** |

> All scores computed using 5-fold cross-validation on the full dataset — no train/test leakage.

---

## Dataset

- **505,206** customer records
- **15 features**: Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction, Gender, Subscription Type, Contract Length
- **Target**: `Churn` (binary) — 55% churn / 45% no churn (balanced)
- Only 1 null row dropped

---

## Project Structure

```
customer-churn-prediction/
├── data/
│   ├── train.csv
│   └── test.csv
├── app/
│   └── app.py          # Streamlit UI
├── Final.ipynb          # Full pipeline notebook
├── model.pkl            # Trained LightGBM pipeline
├── requirements.txt
└── README.md
```

---

## Pipeline

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])),
    ('classifier', LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        random_state=42
    ))
])
```

---

## What Didn't Work

Part of honest ML engineering is documenting dead ends:

- **Feature engineering** (call rate, spend per month, risk flags) — scored 0.9357, worse than baseline. Tree models already discover these interactions through splits.
- **Stacking** (LGBM + RF + Logistic Regression meta-learner) — scored 0.9362. Models make nearly identical errors, leaving no complementary signal to exploit.
- **Conclusion**: 0.936x is the information ceiling of this dataset with these features.

---

## Run Locally

```bash
git clone https://github.com/AhmedAminz/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Requirements

```
pandas
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
joblib
streamlit
```

---

<div align="center">

Built by **Ahmed Amin** — MSC-KFS Data Science Committee

</div>