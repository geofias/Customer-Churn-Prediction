# Customer Churn Prediction

A full-cycle data science project to predict customer churn using machine learning and deep learning models. Includes data extraction via SQL, model building in Python, and interactive business dashboards using Power BI.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Tools & Technologies](#tools--technologies)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Model](#deep-learning-model)
- [Power BI Dashboard](#power-bi-dashboard)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## Project Overview

This project addresses customer churn prediction using structured customer data. It compares traditional machine learning models with deep learning approaches, aiming to provide actionable insights for business stakeholders through an interactive Power BI dashboard.

---

## Objectives

- Use SQL for data extraction and preprocessing.
- Perform EDA and feature engineering with Python.
- Train and evaluate multiple classification models.
- Build and evaluate a simple neural network using TensorFlow/Keras.
- Visualize key business insights and churn risk using Power BI.

---

## Tools & Technologies

| Category        | Tools                                    |
|----------------|-------------------------------------------|
| Languages       | Python, SQL                              |
| Libraries       | pandas, scikit-learn, XGBoost, seaborn   |
| Deep Learning   | TensorFlow, Keras                        |
| Visualization   | matplotlib, seaborn, **Power BI**        |
| Version Control | Git, GitHub                              |

---

## Dataset

- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Features**: Demographics, services used, billing data
- **Target**: `Churn` (Yes/No)

---

## Workflow

### 1. SQL-Based Data Extraction
```sql
SELECT customerID, gender, SeniorCitizen, tenure, MonthlyCharges, TotalCharges
FROM Customers
WHERE Contract = 'Month-to-month';
```

### 2. Data Preprocessing (Python)
- Handle missing values
- One-hot encoding for categorical features
- Standardization of continuous variables

### 3. Exploratory Data Analysis
- Churn rate by gender, contract type
- Distribution plots and heatmaps

## Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost

### Evaluation Metrics:
- Accuracy
- F1 Score
- Precision/Recall
- ROC-AUC
- SHAP/LIME interpretability

## Deep Learning Model
- 3-layer neural network using Keras

```
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

- Binary Crossentropy Loss
- Adam Optimizer
- Epoch tuning for convergence

## Power BI Dashboard
Key visuals created in Power BI include:
- Churn Risk by Tenure Group
- Contract Type vs Churn Rate
- Predicted vs Actual Churn
- Monthly Charges vs Churn Probability

**Link to .pbix file or published report**: [Churn Report](https://www.kaggle.com/blastchar/telco-customer-churn)

## Results
**Insert Table of Models Performance**

## Conclusion
This project demonstrates that both tree-based models and neural networks can effectively predict customer churn. Feature interpretability and business dashboards provide tangible value to non-technical stakeholders.

## Future Work
- Deploy model as an API for live churn scoring
- Include time-based behavioral data
- Improve Power BI dashboard with geographic and revenue metrics
