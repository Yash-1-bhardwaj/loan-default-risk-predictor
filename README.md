# 💰 Loan Default Risk Predictor
# live url: https://loan-default-risk-predictor-jbfyaskzzrkchlyxr4sysf.streamlit.app/
## Overview

This project predicts whether a loan applicant is likely to default on a loan using Machine Learning. The model analyzes applicant information such as income, employment history, loan amount, credit history, and loan characteristics to estimate default risk.

A Streamlit web application is included to allow users to interact with the model through a simple interface.

---

## Features

* Loan default risk prediction
* Default probability estimation
* Interactive Streamlit web application
* Random Forest Classifier
* SHAP Explainability
* Feature Importance Analysis
* ROC-AUC Evaluation
* Joblib Model Deployment

---

## Dataset

The project uses a credit risk dataset containing borrower information including:

* Age
* Income
* Employment Length
* Home Ownership
* Loan Purpose
* Loan Grade
* Loan Amount
* Interest Rate
* Credit History Length
* Previous Default History

Target Variable:

* `loan_status`

  * 0 = No Default
  * 1 = Default

---

## Machine Learning Pipeline

1. Data Cleaning
2. Missing Value Handling
3. Exploratory Data Analysis (EDA)
4. Correlation Analysis
5. Feature Engineering
6. One-Hot Encoding
7. Train-Test Split
8. Random Forest Classification
9. Model Evaluation
10. SHAP Explainability
11. Model Serialization using Joblib
12. Streamlit Deployment

---

## Model Performance

| Metric                    | Score |
| ------------------------- | ----- |
| Accuracy                  | 94%   |
| ROC-AUC                   | 0.93  |
| Precision (Default Class) | 0.98  |
| Recall (Default Class)    | 0.72  |
| F1-Score                  | 0.83  |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SHAP
* Matplotlib
* Joblib
* Streamlit

---

## Running the Project

Clone the repository:

```bash
git clone https://github.com/Yash-1-bhardwaj/loan-default-risk-predictor.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Project Structure

```text
loan-default-risk-predictor/
│
├── app.py
├── loan_default_model.pkl
├── requirements.txt
├── README.md
├── default_risk.ipynb
└── screenshots/
```

---

## Future Improvements

* Hyperparameter Optimization
* Enhanced SHAP Visualizations
* Loan Approval Recommendation System
* Cloud Deployment
* PDF Risk Reports

---

## Author

Yash Bhardwaj

Computer Engineering Student | Machine Learning Enthusiast
<img width="1854" height="860" alt="image" src="https://github.com/user-attachments/assets/90892559-b804-413f-ad2b-a6fdedd2590f" />
<img width="1779" height="835" alt="image" src="https://github.com/user-attachments/assets/e8155ce7-6b24-4d07-8c5d-d51cddcb341f" />
