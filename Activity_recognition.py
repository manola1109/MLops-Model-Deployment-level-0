import streamlit as st
import random
import pandas as pd
import numpy as np
from PIL import Image
from io import StringIO
import joblib
import os

# ----------------------------
# Load model and dependencies
# ----------------------------
base_path = r"D:\Mastering MLOps - Development to Deployment\HumanActivityRecognitionlevel0-230912-124810"

# Load necessary files
train_features = joblib.load(os.path.join(base_path, "Notebook", "model_features", "K12_train_features.joblib"))
activity_model = joblib.load(os.path.join(base_path, "Notebook", "model_results", "model_result_12f_tuned.joblib"))
label_encoder = joblib.load(os.path.join(base_path, "Notebook", "model_features", "encoder_weights.joblib"))
sample_data = pd.read_csv(os.path.join(base_path, "Data", "new_data.csv"))

# Prepare transformed DataFrame for real-time sampling
df_transformed = sample_data[train_features]

# ----------------------------
# Prediction functions
# ----------------------------
def model_real_time_predict(df, model, le):
    prediction = model.predict(df)
    return le.inverse_transform(prediction)

def model_batch_predict(df, features, model, le):
    df.dropna(inplace=True)
    input_df = df[features]
    prediction = model.predict(input_df)
    df['Prediction_label'] = le.inverse_transform(prediction)
    return df

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.title("📱 Human Activity Recognition (Sensor-based)")
st.markdown("Predict physical activity from smartphone sensor data using a trained ML model.")

tab1, tab2 = st.tabs(["📦 Batch Prediction", "⚡ Real-Time Prediction"])

# ----------------------------
# TAB 1: BATCH PREDICTION
# ----------------------------
with tab1:
    st.subheader("Upload CSV for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(batch_df.head())

        if st.button("📊 Predict Batch Activities"):
            result_df = model_batch_predict(batch_df, train_features, activity_model, label_encoder)
            result_counts = pd.DataFrame(result_df['Prediction_label'].value_counts()).rename(columns={'Prediction_label': 'Count'})
            st.bar_chart(result_counts)
            st.write("📋 Prediction Counts:", result_counts)

# ----------------------------
# TAB 2: REAL-TIME PREDICTION
# ----------------------------
with tab2:
    st.subheader("Simulate Real-Time Sensor Input")

    col1, col2 = st.columns(2)

    # Image Display
    with col1:
        image_path = r"D:\Mastering MLOps - Development to Deployment\Streamlit-230831-175522\Streamlit\phone.jpeg"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption="Smartphone Activity Tracker", use_column_width=True)
        else:
            st.warning("📷 Image not found.")

        random_idx = st.slider("Select a random sensor reading", 0, len(df_transformed) - 1)
        sensor_values = df_transformed.iloc[random_idx].values

    # Manual Inputs
    with col2:
        feature_labels = train_features
        input_dict = {}

        for i, label in enumerate(feature_labels):
            input_dict[label] = st.text_input(label, value=str(sensor_values[i]))

        # Run Prediction
        if st.button("🔮 Predict Real-Time Activity"):
            try:
                input_array = np.array([float(v) for v in input_dict.values()]).reshape(1, -1)
                input_df = pd.DataFrame(input_array, columns=feature_labels)
                prediction = model_real_time_predict(input_df, activity_model, label_encoder)
                st.success(f"🧠 Predicted Activity: **{prediction[0]}**")
            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")
