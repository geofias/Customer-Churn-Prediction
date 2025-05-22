# 📦 Customer Churn Prediction

A full-cycle data science project to predict customer churn using machine learning. Includes data extraction via SQL, model building in Python, and interactive business dashboards using Power BI.

---

## 📌 Project Overview
This project focuses on building a machine learning model to predict customer churn using a real-world dataset. The aim is to develop a classifier that can help a company identify customers likely to cancel their subscription, enabling them to take proactive retention actions.

---

## 🎯 Objectives
To develop a predictive model that accurately identifies churned customers using customer behavioral and service-related features. The project uses:
- Feature engineering
- Handling of imbalanced data
- Advanced evaluation metrics
- Threshold optimization
- Visual business insights via a Power BI dashboard

---

## 🛠️ Technologies Used

- Python (Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn)
- SMOTE (imbalanced-learn)
- Power BI (for business-oriented visualization)
- Jupyter Notebook

---

## 🧾 Dataset

- **Source**: [Telco Customer Churn Dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size**: ~7000 rows, multiple features including categorical, numerical, and target (`Churn`)
- **Target variable**: `Churn` (binary: Yes/No)

---

## Workflow

### 1. 🔍 Exploratory Data Analysis (EDA)
- Identified skewed distributions: `TotalCharges` was right-skewed; `tenure` and `MonthlyCharges` showed bimodal patterns.
- Used `np.log1p` to reduce skewness.
- Month-to-month contract type had the most churned customers.
- The majority of customers that churned had an electronic check payment method.
- Churned customers also exhibited the highest monthly charges.

### 2. ⚖️ Data Preprocessing (Python)
- Label Encoding for binary features
- One-hot Encoding for multi-category features
- `SMOTE` to balance minority class (`Churn = Yes`)

### 3. 📈 Modeling
- **Model Used**: `AdaBoostClassifier`
- **Train-Test Split**: 85/15
- **Best Performance on Test Set**:
  - **F1 Score**: 0.62
  - **Recall**: 0.78
  - **Precision**: 0.52
  - **AUC**: 0.83 (based on predicted probabilities)

#### Evaluation Metrics
- Confusion matrix
- Precision, Recall, F1-score
- AUC-ROC curve
- Precision-Recall curve
- Threshold optimization using F1-score as objective

#### Final Confusion Matrix (Test Set)
|            | Predicted No | Predicted Yes |
|------------|--------------|---------------|
| Actual No  | TN = 564     | FP = 207      |
| Actual Yes | FN = 62      | TP = 222      |

### 4. 🧠 Interpretation
- The model favors **recall**, capturing most customers at risk of churning.
- Precision is moderate — acceptable when recall is business-critical.
- The model is **well-calibrated** and **robust**, making it suitable for deployment in a business environment.

---

## 📊 Power BI Dashboard
A Power BI dashboard was created to communicate findings with non-technical stakeholders. It summarizes key insights and model implications in an interactive, visual format.

The Dashboard created in Power BI includes:
- Overview Page
- Customer Profile Analysis
- Tenure & Financial Analysis
- Demographic Analysis

**Link to .pbix file or published report**: [Churn Report](outputs/PowerBI_Report/Churn_PowerBI_Report.pdf)

## 🚀 Next Steps
- Hyperparameter tuning with GridSearchCV
- Add explainability with SHAP values
- Deploy model via Flask or Streamlit
- Share Power BI dashboard via public link or PDF

## 🧾 License
MIT License

---

> **Author**: Tamir Chong  
> **LinkedIn**: [https://www.linkedin.com/in/tamirchong/](https://www.linkedin.com/in/tamirchong/)  
> **Email**: tamir-chong@hotmail.com
