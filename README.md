# 🤖 Machine Learning Projects

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-Projects-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Author:** Ansh Gajera  
**Course:** Machine Learning and Pattern Recognition (MLPR) - Semester 5  
**Student ID:** 23AIML019

---

## 📋 Overview

This repository contains a collection of machine learning projects developed as part of the Machine Learning and Pattern Recognition course. Each project demonstrates different ML techniques, from regression and classification to clustering and anomaly detection.

---

## 📁 Projects

### 1. 💳 Credit Card Fraud Detection
**File:** `Fraud_Detection.ipynb`

A comprehensive approach to detecting fraudulent credit card transactions using machine learning.

**Techniques Used:**
- Data Preprocessing & Feature Engineering
- Handling Imbalanced Data
- Classification Models
- Performance Visualization

**Dataset:** `creditcard.csv` - Contains anonymized transaction records with features V1-V28, Time, Amount, and Class (target)

---

### 2. 📊 Advertisement Budget & Sales Analysis
**File:** `Advertisement_Budget.ipynb`

Analyzing the relationship between advertising budget across different channels and sales performance.

**Techniques Used:**
- Exploratory Data Analysis
- Linear Regression
- Data Visualization
- Correlation Analysis

**Dataset:** `Advertising Budget and Sales.csv`

---

### 3. 🚲 Bike Sharing Demand Prediction
**File:** `BikeSharing.ipynb`

Predicting bike rental count hourly/daily based on environmental and seasonal settings.

**Techniques Used:**
- Linear Regression
- Event and Anomaly Detection
- Time Series Analysis
- Feature Engineering

**Key Features:**
- Weather conditions (temperature, humidity, wind speed)
- Seasonal patterns (season, month, hour)
- Holiday and working day indicators
- Casual vs registered user analysis

---

### 4. 🏥 Diabetes Prediction
**File:** `Diabetes_Prediction.ipynb`

Building a machine learning model to predict whether a person is likely to have diabetes based on health data.

**Techniques Used:**
- K-Nearest Neighbors (KNN) Classifier
- SMOTE for handling class imbalance
- Feature Scaling (StandardScaler)
- Model Evaluation Metrics

**Goal:** Early detection of diabetes to help individuals take preventive steps and manage their health better.

---

### 5. 🍔 Food Delivery User Segmentation
**File:** `FoodDelivery_System.ipynb`

Segmenting users of a food delivery app into meaningful clusters based on behavior and demographics.

**Techniques Used:**
- Unsupervised Learning (Clustering)
- Principal Component Analysis (PCA)
- K-Means Clustering
- Customer Behavior Analysis

**Key Features Analyzed:**
- Age, Total Orders, Average Spend
- App Usage Time
- Favorite Cuisine preferences
- Order frequency patterns

---

### 6. 💰 Health Insurance Charges Prediction
**File:** `Health_Insurance.ipynb`

Predicting medical insurance charges using various regression techniques.

**Techniques Used:**
- OLS (Ordinary Least Squares) Regression
- Linear Regression
- Support Vector Regression (SVR)
- Random Forest Regressor
- Model Comparison

**Dataset Features:**
- Age, BMI, Number of Children
- Smoking Status, Region
- Insurance Charges (target)

---

### 7. 🛒 Online Purchase Intent Prediction
**File:** `Online_Purchase_Intent.ipynb`

Predicting whether a user will complete a purchase during an online shopping session.

**Techniques Used:**
- K-Nearest Neighbors (KNN)
- Logistic Regression, Decision Tree, Random Forest
- SMOTE for Class Balancing
- PCA for Dimensionality Reduction
- Bias-Variance Tradeoff Analysis
- ROC Curve Analysis

**Business Applications:**
- Understanding customer engagement patterns
- Improving personalized targeting and recommendations
- Increasing conversion rates

---

## 🛠️ Technologies & Libraries

| Category | Libraries |
|----------|-----------|
| **Data Manipulation** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Dimensionality Reduction** | PCA |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

### Running the Notebooks
1. Clone the repository
2. Open the desired `.ipynb` file in Jupyter Notebook or VS Code
3. Ensure the required datasets are in the appropriate paths
4. Run cells sequentially

---

## 📊 ML Techniques Covered

| Technique | Projects |
|-----------|----------|
| **Linear Regression** | Advertisement Budget, Bike Sharing, Health Insurance |
| **K-Nearest Neighbors** | Diabetes Prediction, Online Purchase Intent |
| **Random Forest** | Health Insurance, Online Purchase Intent |
| **K-Means Clustering** | Food Delivery System |
| **PCA** | Food Delivery System, Online Purchase Intent |
| **SMOTE** | Diabetes Prediction, Online Purchase Intent, Fraud Detection |
| **Decision Trees** | Online Purchase Intent |
| **SVR** | Health Insurance |

---

## 📈 Project Structure

```
PROJECTS/
├── Advertisement_Budget.ipynb
├── BikeSharing.ipynb
├── Diabetes_Prediction.ipynb
├── FoodDelivery_System.ipynb
├── Fraud_Detection.ipynb
├── Health_Insurance.ipynb
├── Online_Purchase_Intent.ipynb
└── README.md
```

---

## 📝 License

This project is for educational purposes as part of the MLPR course curriculum.

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements or additional projects.

---

## 📧 Contact

**Ansh Gajera**  
Student ID: 23AIML019  
Course: Machine Learning and Pattern Recognition (SEM 5)

---

⭐ **If you find this repository helpful, please consider giving it a star!**
