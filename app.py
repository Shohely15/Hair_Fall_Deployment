import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import re

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
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, label_encoders, feature_names

# Function to clean column names (same as training)
def clean_string(s):
    if pd.isna(s):
        return "Normal"
    # Remove Bengali text in parentheses
    s = re.sub(r'\(.*?\)', '', str(s)).strip()
    return s

# Custom CSS
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
    st.markdown("<h1 class='main-header'>💇 Hair Fall Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center'>Based on Lifestyle Analysis of University Students in Bangladesh</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models
    try:
        model, scaler, label_encoders, feature_names = load_models()
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found! {e}")
        st.info("Please train the model first using the notebook to generate all required files.")
        return
    
    # Create input dictionary with EXACT column names as training
    st.subheader("📋 Personal Information")
    col1, col2 = st.columns(2)
    
    with col1:
        # These names MUST match your training data column names exactly
        age = st.selectbox("Age (বয়স)", ["18–20", "21–23", "24–26", "More than 27"])
        year_study = st.selectbox("Year of Study (কোন বর্ষের ছাত্র/ছাত্রী)", 
                                   ["1st year", "2nd year", "3rd year", "4th year or above"])
        family_history = st.selectbox("Family History of Hair Loss (পরিবারে কারও টাক/চুল পড়ার ইতিহাস আছে কি?)", 
                                       ["Yes", "No", "Not sure"])
        
        st.subheader("💇 Hair Care Habits")
        washing_freq = st.selectbox("Hair Washing Frequency (আপনি সপ্তাহে কয়দিন চুল পরিষ্কার করুন ?)", 
                                     ["0–1 times", "2–3 times", "4–5 times", "6 or more"])
        
        scalp_options = st.multiselect(
            "Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন",
            ["Dandruff (খুশকি)", "Oily scalp (তৈলাক্ত)", "Itching (চুলকানি)", "None"]
        )
        
        chemical_treatment = st.selectbox(
            "Chemical Treatments / Styling (চুলে ডাই, পার্ম, স্ট্রেইটেনিং বা হিট ব্যবহার করেছেন?)",
            ["Never", "Occasionally (মাসে একবারের কম)", "Weekly or more", "Monthly"]
        )
    
    with col2:
        st.subheader("🚶 Lifestyle Factors")
        outdoor_activity = st.selectbox(
            "During your daily travel or outdoor activities, which situation fits you best ( আপনার দৈনন্দিন যাতায়াত বা বাইরের কাজে নিচের কোনটি আপনার সাথে সবচেয়ে বেশি প্রযোজ্য)? ",
            ["No helmet, low exposure", "Pollution only", "Helmet only", "Both helmet & pollution daily", "Headscarf"]
        )
        smoking = st.selectbox("Smoking Habit (ধূমপানের অভ্যাস আছে কি?)", ["Never", "Occasionally"])
        
        st.subheader("🥗 Dietary Habits")
        protein_intake = st.selectbox(
            "Dietary Protein Intake (প্রোটিনযুক্ত খাবার—ডিম/মাছ/মুরগি/ডাল—খাওয়ার হার)",
            ["Never", "Sometimes (সপ্তাহে ২–3 দিন)", "Often (সপ্তাহে ৪–৫ দিন)", "Daily", "Rarely (সপ্তাহে ০–১ দিন)"]
        )
        veg_fruit_intake = st.selectbox(
            "Vegetable/Fruit Intake (সবজি/ফল খাওয়ার অভ্যাস)",
            ["Never", "Sometimes", "Often", "Daily", "Rarely"]
        )
        supplements = st.selectbox("Dietary Supplements (ভিটামিন/আয়রন/বায়োটিন ইত্যাদি সাপ্লিমেন্ট নেন?)", 
                                   ["Never", "Occasionally", "Regularly"])
    
    with st.expander("📊 Additional Information"):
        col3, col4 = st.columns(2)
        
        with col3:
            weight_loss = st.selectbox("Weight Loss Attempts (ওজন কমানোর চেষ্টা করেছেন?)", 
                                        ["Never", "Yes, once", "Yes, multiple times"])
            exercise_freq = st.selectbox(
                "Exercise Frequency (শারীরিক ব্যায়ামের হার)",
                ["Never", "1–2 times/week", "3–4 times/week", "5 or more times/week"]
            )
            stress_level = st.selectbox(
                "Academic/Emotional Stress (পড়াশোনা/ব্যক্তিগত কারণে স্ট্রেস অনুভব করেন?)",
                ["Never", "Rarely", "Sometimes", "Often", "Almost always"]
            )
        
        with col4:
            sleep_duration = st.selectbox("Sleep Duration (গড়ে প্রতিদিন কত ঘণ্টা ঘুমান?)",
                                           ["< 5 hours", "5–6 hours", "7–8 hours", "More than 8 hours"])
            sleep_quality = st.selectbox("Sleep Quality (ঘুমের মান কেমন?)", 
                                          ["Very poor", "Fairly poor", "Fairly good", "Very good"])
            recent_illness = st.selectbox(
                "Recent Illness/COVID History (গত ১ বছরে বড় অসুখ বা কোভিডে আক্রান্ত হয়েছেন?)",
                ["No", "Yes, mild illness", "Yes, severe illness", "Yes, COVID-19"]
            )
    
    predict_button = st.button("🔍 Predict Hair Fall Severity", type="primary", use_container_width=True)
    
    if predict_button:
        # Create DataFrame with EXACT column names as in training
        input_dict = {
            '2.Gender (লিঙ্গ)': ['Female (মহিলা)'],  # Fixed as per your data
            '1. Age (বয়স)': [age],
            '3.Year of Study (কোন বর্ষের ছাত্র/ছাত্রী)': [year_study],
            '4.Family History of Hair Loss (পরিবারে কারও টাক/চুল পড়ার ইতিহাস আছে কি?)': [family_history],
            '5. Current Hair Fall Severity (বর্তমানে আপনার চুল পড়ার মাত্রা কতটা?)': ['Moderate (মাঝারি)'],  # Dummy value
            '6. Hair Washing Frequency (আপনি সপ্তাহে কয়দিন চুল পরিষ্কার করুন   ?)': [washing_freq],
            '7.Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন': [', '.join(scalp_options) if scalp_options else 'None'],
            '8.Chemical Treatments / Styling (চুলে ডাই, পার্ম, স্ট্রেইটেনিং বা হিট ব্যবহার করেছেন?)': [chemical_treatment],
            '9. During your daily travel or outdoor activities, which situation fits you best ( আপনার দৈনন্দিন যাতায়াত বা বাইরের কাজে নিচের কোনটি আপনার সাথে সবচেয়ে বেশি প্রযোজ্য)? ': [outdoor_activity],
            '10.Smoking Habit (ধূমপানের অভ্যাস আছে কি?)': [smoking],
            '11. Dietary Protein Intake (প্রোটিনযুক্ত খাবার—ডিম/মাছ/মুরগি/ডাল—খাওয়ার হার)': [protein_intake],
            '12.Vegetable/Fruit Intake (সবজি/ফল খাওয়ার অভ্যাস)': [veg_fruit_intake],
            '13.Dietary Supplements (ভিটামিন/আয়রন/বায়োটিন ইত্যাদি সাপ্লিমেন্ট নেন?)': [supplements],
            '14.Weight Loss Attempts (ওজন কমানোর চেষ্টা করেছেন?)': [weight_loss],
            '15. Exercise Frequency (শারীরিক ব্যায়ামের হার)': [exercise_freq],
            '16.Academic/Emotional Stress (পড়াশোনা/ব্যক্তিগত কারণে স্ট্রেস অনুভব করেন?)': [stress_level],
            '17.Sleep Duration (গড়ে প্রতিদিন কত ঘণ্টা ঘুমান?)': [sleep_duration],
            '18. Sleep Quality (ঘুমের মান কেমন?)': [sleep_quality],
            '19. Recent Illness/COVID History (গত ১ বছরে বড় অসুখ বা কোভিডে আক্রান্ত হয়েছেন?)': [recent_illness],
            '               SID': [1]  # Dummy SID
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Clean column values (same as training)
        for col in input_df.select_dtypes(include=['object']).columns:
            input_df[col] = input_df[col].apply(clean_string)
        
        # Standardize ranges
        input_df['1. Age (বয়স)'] = input_df['1. Age (বয়স)'].replace({'18–20': '18-20', '21–23': '21-23', '24–26': '24-26'})
        input_df['6. Hair Washing Frequency (আপনি সপ্তাহে কয়দিন চুল পরিষ্কার করুন   ?)'] = input_df['6. Hair Washing Frequency (আপনি সপ্তাহে কয়দিন চুল পরিষ্কার করুন   ?)'].replace({'2–3 times': '2-3 times'})
        
        # Feature Engineering - Create dummy columns for scalp condition
        input_df['Dandruff'] = input_df['7.Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন'].apply(lambda x: 1 if 'Dandruff' in str(x) else 0)
        input_df['Oily_Scalp'] = input_df['7.Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন'].apply(lambda x: 1 if 'Oily' in str(x) else 0)
        input_df['Itching'] = input_df['7.Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন'].apply(lambda x: 1 if 'Itching' in str(x) else 0)
        input_df.drop(columns=['7.Scalp Condition (মাথার ত্বকের অবস্থা) – একাধিক নির্বাচন করুন'], inplace=True)
        
        # Drop target column if present
        if '5. Current Hair Fall Severity (বর্তমানে আপনার চুল পড়ার মাত্রা কতটা?)' in input_df.columns:
            input_df.drop(columns=['5. Current Hair Fall Severity (বর্তমানে আপনার চুল পড়ার মাত্রা কতটা?)'], inplace=True)
        
        # Encode categorical features
        for col in input_df.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError as e:
                    st.error(f"Error encoding {col}: {e}")
                    return
        
        # Ensure all feature names match
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0  # Add missing columns with default value
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale features
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]
        
        # Map prediction to label
        severity_map = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
        severity = severity_map[prediction]
        
        # Display prediction
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
