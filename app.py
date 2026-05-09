import streamlit as st
import pandas as pd
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoders = pickle.load(open("label_encoders.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

st.title("Hair Fall Severity Prediction")

# User Inputs
age = st.text_input("Age")
gender = st.text_input("Gender")
stress = st.text_input("Stress Level")
sleep = st.text_input("Sleep Duration")
protein = st.text_input("Protein Intake")
vegetables = st.text_input("Vegetable/Fruit Intake")
water = st.text_input("Water Intake")
smoking = st.text_input("Smoking Habit")
exercise = st.text_input("Exercise Frequency")

if st.button("Predict"):

    input_dict = {
        feature_names[0]: age,
        feature_names[1]: gender,
        feature_names[2]: stress,
        feature_names[3]: sleep,
        feature_names[4]: protein,
        feature_names[5]: vegetables,
        feature_names[6]: water,
        feature_names[7]: smoking,
        feature_names[8]: exercise
    }

    input_df = pd.DataFrame([input_dict])

    # Encode safely
    for col in input_df.columns:

        if col in label_encoders:

            le = label_encoders[col]

            # Handle unseen labels
            value = str(input_df[col][0])

            if value not in le.classes_:
                st.error(f"Unknown value '{value}' for {col}")
                st.stop()

            input_df[col] = le.transform(input_df[col])

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)[0]

    target_encoder = label_encoders['Current Hair Fall Severity']

    result = target_encoder.inverse_transform([prediction])[0]

    st.success(f"Prediction: {result}")
