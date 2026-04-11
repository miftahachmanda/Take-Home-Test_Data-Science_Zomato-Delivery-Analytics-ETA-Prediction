import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from math import radians, sin, cos, sqrt, atan2

st.set_page_config(page_title="Zomato Delivery Analytics", layout="wide")

st.title("🚚 Zomato Delivery Analytics & ETA Prediction")

st.markdown("## 📌 Business Context")

with st.expander("📖 Lihat Penjelasan Lengkap"):
    st.markdown("""
    ### 🚀 Introduction
    Dalam industri food delivery, akurasi estimasi waktu pengiriman (ETA) menjadi faktor kunci yang mempengaruhi kepuasan dan kepercayaan pelanggan.  

    Melalui analisis dataset Zomato Delivery, dilakukan eksplorasi data dan pengembangan model machine learning untuk:
    - Mengidentifikasi faktor utama yang mempengaruhi waktu pengiriman  
    - Memprediksi durasi delivery secara lebih akurat  
    - Menemukan bottleneck dalam proses operasional  

    👉 Hasil analisis ini digunakan untuk mendukung:
    - Optimasi alokasi dan penjadwalan driver  
    - Pengurangan keterlambatan pengiriman  
    - Peningkatan akurasi estimasi waktu (ETA)  

    ---
    ### ⚠️ Problem Statement
    Waktu pengiriman merupakan salah satu faktor utama yang menentukan kualitas layanan dalam industri food delivery.  

    Keterlambatan pengiriman tidak hanya menurunkan kepuasan pelanggan, tetapi juga meningkatkan risiko komplain dan churn.  
    Selain itu, estimasi waktu pengiriman (ETA) yang tidak akurat dapat menciptakan ekspektasi yang tidak realistis.  

    Kompleksitas operasional seperti:
    - Jarak pengiriman  
    - Kondisi lalu lintas  
    - Cuaca  
    - Jumlah pengiriman dalam satu perjalanan  

    turut mempengaruhi durasi delivery. Oleh karena itu, diperlukan pendekatan berbasis data untuk meningkatkan efisiensi operasional.

    ---
    ### 🎯 Tujuan
    - Mengidentifikasi faktor utama yang mempengaruhi waktu delivery  
    - Menemukan bottleneck dalam proses pengiriman  
    - Membangun model prediksi ETA yang lebih akurat  
    - Mengoptimalkan alokasi dan penjadwalan driver  

    ---
    ### 💡 Dampak Bisnis
    - Meningkatkan kepuasan pelanggan  
    - Mengurangi churn  
    - Mendukung pengambilan keputusan berbasis data  
    """)
    
# ========================
# Upload Data
# ========================
uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # ========================
    # PREPROCESSING
    # ========================
    st.subheader("⚙️ Data Preprocessing")

    df.rename(columns={'Time_Orderd': 'Time_Ordered'}, inplace=True)
    df.replace('NaN ', np.nan, inplace=True)

    # Fill missing
    num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = ['Weather_conditions', 'Road_traffic_density', 'Festival', 'City']
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=['Time_Ordered'])

    st.success("Preprocessing selesai ✅")

    # ========================
    # FEATURE ENGINEERING
    # ========================
    st.subheader("🧪 Feature Engineering")

    df['Time_Ordered'] = pd.to_datetime(df['Time_Ordered'], errors='coerce')
    df['Time_Order_picked'] = pd.to_datetime(df['Time_Order_picked'], errors='coerce')

    df = df.dropna(subset=['Time_Ordered','Time_Order_picked'])

    df['prep_time'] = (df['Time_Order_picked'] - df['Time_Ordered']).dt.total_seconds() / 60
    df['prep_time'] = df['prep_time'].apply(lambda x: x if x > 0 else x + 1440)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    df['distance_km'] = df.apply(lambda row: haversine(
        row['Restaurant_latitude'],
        row['Restaurant_longitude'],
        row['Delivery_location_latitude'],
        row['Delivery_location_longitude']
    ), axis=1)

    df['distance_category'] = pd.cut(
        df['distance_km'],
        bins=[0,2,5,10,20],
        labels=['Very Near','Near','Medium','Far']
    )

    st.success("Feature engineering selesai 🚀")

    # ========================
    # EDA
    # ========================
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Distribusi Delivery Time")
        fig, ax = plt.subplots()
        sns.histplot(df['Time_taken (min)'], bins=30, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Distance vs Time")
        dist = df.groupby('distance_category')['Time_taken (min)'].mean()
        fig, ax = plt.subplots()
        dist.plot(kind='line', marker='o', ax=ax)
        st.pyplot(fig)

    # ========================
    # LOAD MODEL
    # ========================
    st.subheader("🤖 ETA Prediction")

    try:
        model = joblib.load("best_random_forest_model.joblib")
    except:
        st.warning("Model belum tersedia, upload model terlebih dahulu")
        model_file = st.file_uploader("Upload Model (.joblib)")
        if model_file:
            model = joblib.load(model_file)
        else:
            model = None

    if model:

        st.markdown("### Input Data untuk Prediksi")

        age = st.slider("Driver Age", 18, 50, 25)
        rating = st.slider("Rating", 1.0, 5.0, 4.0)
        distance = st.slider("Distance (km)", 0.5, 20.0, 5.0)
        traffic = st.selectbox("Traffic", df['Road_traffic_density'].unique())
        weather = st.selectbox("Weather", df['Weather_conditions'].unique())
        multiple = st.selectbox("Multiple Deliveries", [0,1,2,3])
        vehicle = st.slider("Vehicle Condition", 0, 5, 3)

        if st.button("Predict ETA"):

            input_df = pd.DataFrame({
                'Delivery_person_Age':[age],
                'Delivery_person_Ratings':[rating],
                'distance_km':[distance],
                'Road_traffic_density':[traffic],
                'Weather_conditions':[weather],
                'multiple_deliveries':[multiple],
                'Vehicle_condition':[vehicle]
            })

            # NOTE: encoding harus sama seperti training
            input_df = pd.get_dummies(input_df)

            # align kolom
            model_features = model.feature_names_in_
            input_df = input_df.reindex(columns=model_features, fill_value=0)

            pred = model.predict(input_df)[0]

            st.success(f"⏱️ Estimasi Waktu Delivery: {round(pred,2)} menit")

    # ========================
    # INSIGHT
    # ========================
    st.subheader("💡 Business Insight")

    st.markdown("""
    - Jarak adalah faktor paling berpengaruh
    - Traffic & cuaca memperlambat delivery
    - Multiple deliveries meningkatkan waktu
    - Driver rating tinggi → lebih cepat
    """)

else:
    st.info("Silakan upload dataset terlebih dahulu")