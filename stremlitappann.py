import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('model.h5', compile=False)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_gender = pickle.load(f)
    with open('one_hot_encoder_geo.pkl', 'rb') as f:
        one_hot_geo = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, label_gender, one_hot_geo, scaler

model, label_gender, one_hot_geo, scaler = load_artifacts()  


st.title("Bank Customer Churn Prediction")
st.write("Enter customer details to predict likelihood of churn.")

credit_score = st.number_input("Credit Score", min_value=300, max_value=792, value=720)
gender = st.selectbox("Gender", label_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
tenure = st.number_input("Tenure (Years with Bank)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, value=75000.0, step=100.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, value=85000.0, step=100.0)
geography = st.selectbox("Geography", one_hot_geo.categories_[0])

if st.button("Predict Churn"):
    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": label_gender.transform([gender]),
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active],
        "EstimatedSalary": [salary],
        "Geography": [geography]
    })

    # One-hot encode Geography
    geo_encoded = one_hot_geo.transform(input_data[['Geography']])
    geo_df = pd.DataFrame(geo_encoded,
                          columns=one_hot_geo.get_feature_names_out(['Geography']),
                          index=input_data.index)

    input_data = pd.concat([input_data.drop(columns=['Geography']), geo_df], axis=1)

    # Scale features
    scaled_input = scaler.transform(input_data)

    # Predict probability
    prob = float(model.predict(scaled_input)[0][0])

    # Set threshold
    threshold = 0.37  # from the best F1 score

    if prob >= threshold:
        confidence = prob * 100
        st.error(f"Customer is likely to CHURN (Confidence: {confidence:.2f}%)")
    else:
        confidence = (1 - prob) * 100
        st.success(f"Customer is likely to NOT CHURN (Confidence: {confidence:.2f}%)")

    st.write(f"**Churn Probability:** {prob*100:.2f}%")


# Explanation
with st.expander("How this works"):
    st.write("""
    This project uses the **Churn Modelling** dataset to predict whether a bank customer will exit (churn) based on demographic, account, and activity details.
    """)
