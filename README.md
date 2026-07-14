# 💳 Credit Card Approval Prediction

Predict credit card approval using Machine Learning.

## 📌 Problem Statement

Banks receive thousands of credit card applications daily. Manual review is slow and error-prone. This project automates approval decisions using ML.

## 🎯 Objectives

- Build a predictive model for credit card approval
- Create a web application for real-time predictions
- Compare multiple ML algorithms

## 📊 Dataset

| Feature | Description |
| Gender | 1=Male, 0=Female |
| Age | Applicant age |
| Debt | Existing debt |
| Married | Marital status |
| Industry | Work sector |
| CreditScore | Credit rating |
| Approved | Target (1=Approved, 0=Rejected) |

**Size:** 690 records, 16 features

## 🔧 Tech Stack

- Python 3.x
- Flask (Web Framework)
- Scikit-learn (ML)
- Pandas & NumPy (Data Processing)

## 🚀 Installation

```bash
git clone https://github.com/243ta04037-induja/Credit_Card_Approval_Prediction
cd Credit-Card-Approval-Prediction
pip install -r requirements.txt
python model/train.py
python app.py
