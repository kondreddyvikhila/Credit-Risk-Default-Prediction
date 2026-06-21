# Credit Risk Default Prediction System
Machine Learning-based Credit Risk Assessment using XGBoost, SHAP Explainability, and Streamlit Deployment.
## Project Overview

This project develops a Machine Learning-based Credit Risk Default Prediction System to identify customers who are likely to default on loans. The system assists financial institutions in making informed lending decisions by analyzing customer credit-related information and predicting default risk.

The project includes data preprocessing, feature engineering, class balancing using SMOTE, model development, hyperparameter tuning, model explainability using SHAP, and deployment using Streamlit.

---

## Live Demo

### Streamlit Application

https://credit-risk-default-prediction-jae6pha7rbbvkqdtk3mat3.streamlit.app/

### GitHub Repository

https://github.com/kondreddyvikhila/Credit-Risk-Default-Prediction

---

## Dataset Information

* Total Records: 5000
* Total Attributes: 42
* Features Used: 40
* Target Variable: `default_12m`
* Class Imbalance handled using SMOTE

---

## Methodology

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Data Preprocessing
5. Train-Test Split
6. SMOTE for Class Balancing
7. Model Building
8. Hyperparameter Tuning
9. Cross Validation
10. SHAP Explainability
11. Streamlit Deployment

---

## Models Developed

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 75.8%    | 43.97%    | 76.5%  | 55.84%   | 98.51%  |
| Random Forest       | 91.1%    | 78.46%    | 76.5%  | 77.47%   | 95.33%  |
| XGBoost             | 93.6%    | 83.01%    | 85.5%  | 84.24%   | 98.28%  |
| Tuned XGBoost       | 94.7%    | 85.51%    | 88.5%  | 86.98%   | 98.54%  |

### Best Model

**Tuned XGBoost**

* Accuracy: 94.7%
* ROC-AUC: 98.54%
* Cross Validation Score: 99.64%

---

## Hyperparameter Tuning

Best Parameters:

```python
learning_rate = 0.1
max_depth = 5
n_estimators = 200
subsample = 0.8
```

---

## Key Features

* Credit Risk Prediction
* Default Probability Estimation
* Risk Categorization
* SHAP Explainability
* Feature Importance Visualization
* Real-Time Streamlit Deployment

---

## Business Insights

* Customers with high debt-to-income ratios are more likely to default.
* Previous defaults significantly increase default risk.
* High Average DPD is a strong predictor of loan default.
* Credit utilization strongly influences repayment behavior.
* Behavioral credit features improve risk assessment accuracy.

---

## Project Structure

```text
Credit-Risk-Default-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│
├── images/
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── streamlit_dashboard.png
│
├── reports/
│   ├── Credit_Risk_Project_Report.docx
│   └── Credit_Risk_Project_Report.pdf
│
└── Credit_Risk_Prediction.ipynb
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Matplotlib
* Seaborn
* Streamlit

---

## Future Scope

* Integration with real-world banking datasets.
* Real-time database connectivity.
* Automated loan approval systems.
* Advanced ensemble learning techniques.
* Continuous model retraining using new customer data.

---

## Author

**Kondreddy Vikhila**

Machine Learning Internship Project

