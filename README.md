# 🚢 Titanic Survival Prediction – End-to-End ML Pipeline

## 📌 Overview

This project builds a **production-style machine learning pipeline** to predict passenger survival on the Titanic dataset.

Instead of just training a model, this project focuses on:

* Structured preprocessing
* Preventing data leakage
* Clean feature engineering
* Scalable pipeline architecture
* Reproducible ML workflow

The goal was to implement how ML systems are built in real-world environments — not just inside notebooks.

---

## 🎯 Problem Statement

Predict whether a passenger survived the Titanic disaster based on structured features such as:

* Passenger Class (Pclass)
* Gender (Sex)
* Age
* Fare
* Number of siblings/spouses (SibSp)
* Number of parents/children (Parch)
* Embarked Port

**Target Variable:**
`Survived` → (0 = No, 1 = Yes)

---

## 🛠️ Tech Stack

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

Core ML Components Used:

* `Pipeline`
* `ColumnTransformer`
* `StandardScaler`
* `OneHotEncoder`
* Logistic Regression
* Decision Tree
* Random Forest
* Train-Test Split
* Classification Metrics

---

## 🏗️ Project Architecture

```
titanic-survival-pipeline/
│
├── titanicpipleline.ipynb
├── README.md
└── dataset/
```

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Understanding

* Inspected missing values
* Identified categorical vs numerical features
* Checked class distribution

---

### 2️⃣ Data Cleaning

* Handled missing values in `Age` and `Embarked`
* Removed non-informative columns (if applicable)
* Structured feature groups for transformation

---

### 3️⃣ Feature Engineering

Numerical Features:

* Imputation
* Standardization using `StandardScaler`

Categorical Features:

* Imputation
* One-Hot Encoding using `OneHotEncoder`

All transformations handled using:

```
ColumnTransformer + Pipeline
```

This ensures:

* No data leakage
* Clean transformation workflow
* Reproducibility
* Scalability

---

### 4️⃣ Model Training

Implemented multiple models:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Compared model performance using:

* Accuracy Score
* Confusion Matrix
* Precision
* Recall
* F1 Score

---

### 5️⃣ Model Evaluation

* Checked overfitting using train vs test performance
* Evaluated classification quality beyond just accuracy
* Analyzed confusion matrix for survival prediction bias

---

## 📊 Key Insights

* Gender and Passenger Class were strong survival predictors.
* Higher fare and first-class passengers had better survival probability.
* Proper preprocessing significantly improved model stability.

---

## 🚀 How to Run

### 1. Clone Repository

```
git clone https://github.com/mehvishsheikh31/titanic-survival-pipeline.git
```

### 2. Install Dependencies

```
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 3. Run Notebook

```
jupyter notebook titanicpipleline.ipynb
```

---

## 🔥 What This Project Demonstrates

* Understanding of ML pipeline architecture
* Feature preprocessing using industry-standard tools
* Awareness of data leakage risks
* Model comparison & evaluation skills
* Clean and structured ML workflow

This project moves beyond beginner-level “train and print accuracy” implementations.

---

## 🚀 Future Improvements

* Cross-validation
* Hyperparameter tuning using GridSearchCV
* Model persistence using joblib
* Deployment with Flask or Streamlit
* CI/CD integration
* Dockerization


# titanic_survival
