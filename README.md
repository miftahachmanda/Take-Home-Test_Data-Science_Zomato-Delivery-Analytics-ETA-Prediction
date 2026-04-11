# Take-Home-Test_Data-Science_Zomato-Delivery-Analytics-ETA-Prediction
End-to-end data science project to analyze and predict food delivery time (ETA) using machine learning, with interactive dashboard built in Streamlit.

# 🚚 Zomato Delivery Analytics & ETA Prediction

An end-to-end data science project focused on analyzing and predicting food delivery time (ETA) using machine learning. This project combines data preprocessing, exploratory data analysis (EDA), feature engineering, and predictive modeling, deployed through an interactive Streamlit dashboard.

---

## 📌 Business Context

In the food delivery industry, accurate Estimated Time of Arrival (ETA) plays a crucial role in customer satisfaction and trust. Delays or inaccurate estimates can lead to complaints, churn, and poor user experience.

This project aims to:
- Identify key factors affecting delivery time  
- Predict delivery duration more accurately  
- Detect operational bottlenecks  

---

## ⚠️ Problem Statement

Delivery time is influenced by multiple operational factors such as:
- Delivery distance  
- Traffic conditions  
- Weather conditions  
- Number of deliveries per trip  

Inaccurate ETA predictions can create unrealistic expectations and negatively impact customer satisfaction. Therefore, a data-driven approach is needed to improve prediction accuracy and operational efficiency.

---

## 🎯 Objectives

- Analyze factors influencing delivery time  
- Build a predictive model for ETA  
- Improve delivery efficiency and reduce delays  
- Support data-driven decision making  

---

## 📊 Dataset

The dataset includes:
- Delivery personnel information  
- Order and pickup timestamps  
- Traffic & weather conditions  
- Location coordinates  
- Delivery duration  

---

## ⚙️ Tech Stack

- Python (Pandas, NumPy)
- Visualization (Matplotlib, Seaborn)
- Machine Learning (Scikit-learn)
- Deployment (Streamlit)

---

## 🔍 Key Insights

- 📍 Distance is the most influential factor in delivery time  
- 🚦 Heavy traffic significantly increases delivery duration  
- 🌧️ Bad weather leads to longer and more variable delivery times  
- 📦 Multiple deliveries increase complexity and delay  
- ⭐ Higher-rated drivers tend to deliver faster  

---

## 🤖 Model Performance

| Model              | MAE  | RMSE | R²   |
|-------------------|------|------|------|
| Linear Regression | 4.63 | 5.80 | 0.61 |
| Random Forest     | 3.13 | 3.91 | 0.82 |

✅ Random Forest selected as the best model  

---

## 🚀 Features

- Data preprocessing & cleaning  
- Feature engineering (distance, prep time, etc.)  
- Interactive visualizations  
- ETA prediction using trained ML model  
- User-friendly dashboard with Streamlit  

---

## 🖥️ Demo (Local)

```bash
pip install -r requirements.txt
streamlit run app.py

## ⚠️ Model Setup Instructions

The trained model file is not included in this repository due to file size limitations.

To run the Streamlit application properly, please follow these steps:

1. Run the notebook file (`.ipynb`) to train the model  
2. Save the trained model as `.pkl`  
3. Download the generated model file  
4. Place the model file inside the same folder as the Streamlit app (`app.py`)  

Make sure the model file name matches the one used in the code (e.g., `best_model.pkl`).

Without the model file, the prediction feature will not work.
