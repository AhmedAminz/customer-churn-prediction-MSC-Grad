# Customer Churn Prediction

A machine learning project to predict customer churn using classification models.

## Dataset
- `data/train.csv` — training set (~404k samples)
- `data/test.csv` — test set (~100k samples)
- 15 features: demographic, behavioral, and subscription data
- Target: `Churn` (binary, balanced ~55/45)

## Workflow
1. **Preprocessing** — null handling, float-to-int casting, dropping CustomerID, encoding categoricals with OneHotEncoder, scaling numericals with StandardScaler via ColumnTransformer pipeline
2. **Modeling** — trained and compared Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
3. **Evaluation** — 5-fold cross-validation on the full dataset to prevent data leakage
4. **Optimization** — hyperparameter tuning on best performing model

## Results

| Model | CV Accuracy |
|---|---|
| Logistic Regression | 0.840 |
| Gradient Boosting | 0.913 |
| XGBoost | 0.933 |
| Random Forest | 0.934 |
| **LightGBM (tuned)** | **0.936** |

## Requirements

pip install -r requirements.txt


## Usage
Open `Final.ipynb` and run all cells sequentially. Dataset must be in the `data/` folder.


## App Link

https://customer-churn-prediction-msc-grad-2hchzdkrpcugri87ksrcc5.streamlit.app/
