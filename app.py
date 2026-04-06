import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("Water_Potability_model.pkl")
scaler = joblib.load("scaler.pkl")

# App Title
st.title("💧 Water Potability Prediction App")
st.write("Enter the water quality parameters to predict if water is safe to drink")

# Input fields (9 features)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0)

hardness = st.number_input("Hardness", min_value=0.0, value=200.0)

solids = st.number_input("Total Dissolved Salts (TDS)", min_value=0.0, value=20000.0)

chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)

sulphate = st.number_input("Sulphate", min_value=0.0, value=330.0)

conductivity = st.number_input("Conductivity", min_value=0.0, value=420.0)

organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=14.0)

trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=66.0)

turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# Predict Button
if st.button("Predict Water Potability"):

    # Convert input to array
    input_data = np.array([[ph, hardness, solids, chloramines,
                            sulphate, conductivity, organic_carbon,
                            trihalomethanes, turbidity]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Output Result
    if prediction[0] == 1:
        st.success("✅ The Water is Potable (Safe to Drink).")
    else:
        st.error("❌ The Water is Not Potable (Unsafe to Drink).")


