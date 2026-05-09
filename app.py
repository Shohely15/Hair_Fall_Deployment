import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Hair Fall Prediction System",
    page_icon="💇",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    return model, scaler, label_encoders

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E4053;
        padding: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .severe {
        background-color: #FF6B6B;
        color: white;
    }
    .moderate {
        background-color: #FFB347;
        color: white;
    }
    .mild {
        background-color: #4ECDC4;
        color: white;
    }
    .none {
        background-color: #95E77E;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 class='main-header'>💇 Hair Fall Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Based on Lifestyle Analysis of University Students in Bangladesh</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    try:
        model, scaler, label_encoders = load_models()
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please train the model first.")
        st.info("Run the training notebook to generate model.pkl, scaler.pkl, and label_encoders.pkl")
        return
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        age = st.selectbox("Age", ["18-20", "21-23", "24-26", "More than 27"])
        year_of_study = st.selectbox("Year of Study", ["1st year", "2nd year", "3rd year", "4th year or above"])
        family_history = st.selectbox("Family History of Hair Loss", ["Yes", "No", "Not sure"])
        
        st.subheader("💇 Hair Care Habits")
        washing_freq = st.selectbox("Hair Washing Frequency", ["0–1 times", "2–3 times", "4–5 times", "6 or more"])
        
        scalp_condition = st.multiselect(
            "Scalp Condition (Select all that apply)",
            ["Dandruff", "Oily scalp", "Itching", "None"]
        )
        chemical_treatment = st.selectbox(
            "Chemical Treatments / Styling",
            ["Never", "Occasionally", "Weekly or more", "Monthly"]
        )
    
    with col2:
        st.subheader("🚶 Lifestyle Factors")
        outdoor_activity = st.selectbox(
            "Daily Travel/Outdoor Activity",
            ["No helmet, low exposure", "Pollution only", "Helmet only", "Both helmet & pollution daily", "Headscarf"]
        )
        smoking = st.selectbox("Smoking Habit", ["Never", "Occasionally"])
        
        st.subheader("🥗 Dietary Habits")
        protein_intake = st.selectbox(
            "Dietary Protein Intake",
            ["Never", "Sometimes (সপ্তাহে ২–3 দিন)", "Often (সপ্তাহে ৪–৫ দিন)", "Daily"]
        )
        veg_fruit_intake = st.selectbox(
            "Vegetable/Fruit Intake",
            ["Never", "Sometimes", "Often", "Daily", "Rarely"]
        )
        supplements = st.selectbox("Dietary Supplements", ["Never", "Occasionally", "Regularly"])
    
    # More inputs in expandable section
    with st.expander("📊 Additional Information"):
        col3, col4 = st.columns(2)
        
        with col3:
            weight_loss = st.selectbox("Weight Loss Attempts", ["Never", "Yes, once", "Yes, multiple times"])
            exercise_freq = st.selectbox(
                "Exercise Frequency",
                ["Never", "1–2 times/week", "3–4 times/week", "5 or more times/week"]
            )
            stress_level = st.selectbox(
                "Academic/Emotional Stress",
                ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            )
        
        with col4:
            sleep_duration = st.selectbox(
                "Sleep Duration",
                ["< 5 hours", "5–6 hours", "7–8 hours", "More than 8 hours"]
            )
            sleep_quality = st.selectbox("Sleep Quality", ["Very poor", "Fairly poor", "Fairly good", "Very good"])
            recent_illness = st.selectbox(
                "Recent Illness/COVID History",
                ["No", "Yes, mild illness", "Yes, severe illness", "Yes, COVID-19"]
            )
    
    # Prediction button
    st.markdown("---")
    predict_button = st.button("🔍 Predict Hair Fall Severity", type="primary", use_container_width=True)
    
    if predict_button:
        # Prepare input data
        input_data = {
            'Age': age,
            'Year of Study': year_of_study,
            'Family History of Hair Loss': family_history,
            'Hair Washing Frequency': washing_freq,
            'Scalp Condition': ', '.join(scalp_condition) if scalp_condition else 'None',
            'Chemical Treatments / Styling': chemical_treatment,
            'During your daily travel or outdoor activities, which situation fits you best': outdoor_activity,
            'Smoking Habit': smoking,
            'Dietary Protein Intake': protein_intake,
            'Vegetable/Fruit Intake': veg_fruit_intake,
            'Dietary Supplements': supplements,
            'Weight Loss Attempts': weight_loss,
            'Exercise Frequency': exercise_freq,
            'Academic/Emotional Stress': stress_level,
            'Sleep Duration': sleep_duration,
            'Sleep Quality': sleep_quality,
            'Recent Illness/COVID History': recent_illness
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Feature Engineering - Create dummy columns for scalp condition
        input_df['Dandruff'] = input_df['Scalp Condition'].apply(lambda x: 1 if 'Dandruff' in str(x) else 0)
        input_df['Oily_Scalp'] = input_df['Scalp Condition'].apply(lambda x: 1 if 'Oily' in str(x) else 0)
        input_df['Itching'] = input_df['Scalp Condition'].apply(lambda x: 1 if 'Itching' in str(x) else 0)
        input_df.drop(columns=['Scalp Condition'], inplace=True)
        
        # Encode categorical features
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    st.error(f"Invalid value for {col}")
                    return
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Map prediction to label
        severity_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
        severity = severity_map[prediction]
        
        # Display prediction with color coding
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        if severity == "Severe":
            st.markdown(f"""
            <div class='prediction-box severe'>
                <h2>⚠️ Severe Hair Fall</h2>
                <p>Your lifestyle factors indicate a high risk of severe hair fall.</p>
                <p>Confidence: {prediction_proba[prediction]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("**Recommendations:** Consult a dermatologist immediately. Consider lifestyle changes.")
            
        elif severity == "Moderate":
            st.markdown(f"""
            <div class='prediction-box moderate'>
                <h2>⚠️ Moderate Hair Fall</h2>
                <p>Your lifestyle factors indicate moderate hair fall risk.</p>
                <p>Confidence: {prediction_proba[prediction]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.info("**Recommendations:** Improve diet, manage stress, and consider hair care products.")
            
        elif severity == "Mild":
            st.markdown(f"""
            <div class='prediction-box mild'>
                <h2>✅ Mild Hair Fall</h2>
                <p>Your lifestyle factors indicate low hair fall risk.</p>
                <p>Confidence: {prediction_proba[prediction]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("**Recommendations:** Maintain healthy habits. Regular check-ups recommended.")
            
        else:
            st.markdown(f"""
            <div class='prediction-box none'>
                <h2>✅ No Hair Fall</h2>
                <p>Your lifestyle factors indicate no significant hair fall risk.</p>
                <p>Confidence: {prediction_proba[prediction]*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("**Recommendations:** Continue healthy lifestyle habits!")
        
        # Show probability distribution
        st.subheader("📈 Probability Distribution")
        prob_df = pd.DataFrame({
            'Severity': ['None', 'Mild', 'Moderate', 'Severe'],
            'Probability': prediction_proba
        })
        st.bar_chart(prob_df.set_index('Severity'))

if __name__ == "__main__":
    main()
