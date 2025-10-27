# 🧠 Customer Churn Prediction – End-to-End Machine Learning Project

An end-to-end Customer Churn Prediction system built with robust data science practices, including EDA, feature engineering, class imbalance correction (SMOTE), multiple supervised learning models, MLflow tracking, and Gradio deployment for interactive prediction.

## 🚀 Project Overview

The goal of this project is to predict whether a customer will churn (exit) from a bank’s service based on demographic and behavioral factors.
It follows the complete ML lifecycle, from raw data to deployment, ensuring reproducibility and experiment management using MLflow.

Key Highlights

## 📊 Comprehensive Exploratory Data Analysis (EDA)

⚖️ Data balancing using SMOTE

🧠 Model training with 7 algorithms

📈 Automated experiment logging with MLflow

🌐 User-friendly Gradio interface for real-time predictions

## 🧰 Tech Stack
Category	Tools / Libraries
Language	Python 3.10+
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn, XGBoost, Imbalanced-learn
Experiment Tracking	MLflow
Deployment	Gradio
Environment	VS Code

## 📁 Project Structure

EndToEnd-Churn/
│
├── EndToEnd.ipynb               # Main notebook (EDA + Model Training + Deployment)
├── my_mlflow_utils.py           # Custom MLflow tracking utility
├── requirements.txt             # Dependencies
├── models/                      # Serialized trained models
├── dataset/
│   └── Churn_Modelling.csv      # Input dataset
└── README.md                    # Documentation

## 📊 Dataset Description

Source: Churn_Modelling.csv (10,000 rows × 14 columns)
Objective: Predict the column Exited (1 = churned, 0 = retained)

## Feature	Description
CreditScore	Customer’s credit rating
Geography	Country of residence
Gender	Male / Female
Age	Age in years
Tenure	Number of years as customer
Balance	Bank account balance
NumOfProducts	Number of bank products held
HasCrCard	Whether the customer has a credit card
IsActiveMember	Active membership indicator
EstimatedSalary	Annual income estimate
Exited	Target variable (1 = churned)
⚙️ Model Training & Results

The following models were trained and evaluated on the preprocessed dataset.
SMOTE was applied to address class imbalance before training.
Each experiment was tracked via MLflow for versioning and reproducibility.

## Model	Accuracy
Random Forest Classifier	86.2%
XGBoost Classifier	81.7%
Gradient Boosting Classifier	82.6%
Decision Tree Classifier	80.7%
K-Nearest Neighbors (KNN)	76.4%
Logistic Regression	73.4%
Support Vector Machine (SVM – RBF)	45.9%

✅ Random Forest achieved the highest accuracy (86.2%), demonstrating strong generalization performance.
⚠️ SVM performed poorly due to non-linear class separability and scaling sensitivity.

## 🧪 Experiment Tracking with MLflow

All training runs and metrics are logged in MLflow, including:

Model name and parameters

Accuracy, Precision, Recall, F1-Score

Confusion matrix

Stored .pkl model artifacts

Run MLflow locally:
mlflow ui


Then open http://localhost:5000
 in your browser.

## 🌐 Deployment via Gradio

A Gradio web interface allows users to input customer details and instantly receive churn predictions.

Run the App:
python app.py


or from within the notebook:

import gradio as gr
# interface = gr.Interface(...)
# interface.launch()


# Demo Link (Example):
🔗 Live Gradio App
 (replace with your actual link)

## 🧩 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/EndToEnd-Churn.git
cd EndToEnd-Churn


## 3️⃣ Install Dependencies
pip install -r requirements.txt

## 4️⃣ Run the Notebook or App
jupyter notebook EndToEnd.ipynb
# or
python app.py
