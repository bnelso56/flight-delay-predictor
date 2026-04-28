from datetime import date, time
from pathlib import Path
import math

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin


class FlightPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor that transforms user inputs into model features"""
    
    def __init__(self):
        self.carrier_categories = None
        self.dest_freq_dict = None
        self.origin_freq_dict = None
        
    def fit(self, X, y=None):
        """Learn carrier categories and airport frequency mappings from training data"""
        self.carrier_categories = sorted(X['CARRIER'].unique())
        self.dest_freq_dict = X['DEST'].value_counts(normalize=True).to_dict()
        self.origin_freq_dict = X['ORIGIN'].value_counts(normalize=True).to_dict()
        return self
    
    def transform(self, X):
        """Transform user inputs to model features"""
        X_copy = X.copy()
        
        # Extract basic inputs
        X_copy['time_in_hours'] = X_copy['time_in_hours']
        X_copy['DAY_WEEK'] = X_copy['DAY_WEEK']
        
        # Add day of month from the flight date
        X_copy['DAY_OF_MONTH'] = X_copy.get('DAY_OF_MONTH', 15)
        
        # Add distance with a default value
        X_copy['DISTANCE'] = X_copy.get('DISTANCE', 200)
        
        # Rename WEATHER to Weather (capital W for model compatibility)
        X_copy['Weather'] = X_copy['WEATHER']
        
        # Create cyclical features
        X_copy['dow_sin'] = np.sin(2 * np.pi * X_copy['DAY_WEEK'] / 7)
        X_copy['dow_cos'] = np.cos(2 * np.pi * X_copy['DAY_WEEK'] / 7)
        
        X_copy['time_sin'] = np.sin(2 * np.pi * X_copy['time_in_hours'] / 24)
        X_copy['time_cos'] = np.cos(2 * np.pi * X_copy['time_in_hours'] / 24)
        
        X_copy['is_weekend'] = (X_copy['DAY_WEEK'] >= 5).astype(int)
        
        # Create frequency features
        mean_freq = 1 / (len(self.dest_freq_dict) + 1)
        X_copy['DEST_freq'] = X_copy['DEST'].map(self.dest_freq_dict).fillna(mean_freq)
        X_copy['ORIGIN_freq'] = X_copy['ORIGIN'].map(self.origin_freq_dict).fillna(mean_freq)
        
        # One-hot encode carrier (no drop_first to match training data with all 7 carriers)
        carrier_dummies = pd.get_dummies(X_copy['CARRIER'], prefix='CARRIER', dtype=int)
        
        # Ensure all carrier columns exist (initialize missing ones to 0)
        all_carriers = ['CARRIER_DH', 'CARRIER_DL', 'CARRIER_MQ', 'CARRIER_OH', 'CARRIER_RU', 'CARRIER_UA', 'CARRIER_US']
        for carrier in all_carriers:
            if carrier not in carrier_dummies.columns:
                carrier_dummies[carrier] = 0
        
        # Build feature matrix in exact training order
        feature_order = ['DISTANCE', 'Weather', 'DAY_OF_MONTH', 'dow_sin', 'dow_cos', 'time_sin', 'time_cos', 
                        'is_weekend', 'CARRIER_DH', 'CARRIER_DL', 'CARRIER_MQ', 'CARRIER_OH', 'CARRIER_RU', 
                        'CARRIER_UA', 'CARRIER_US', 'DEST_freq', 'ORIGIN_freq']
        
        # Create base features
        X_final = X_copy[['DISTANCE', 'Weather', 'DAY_OF_MONTH', 'dow_sin', 'dow_cos', 'time_sin', 'time_cos', 'is_weekend', 'DEST_freq', 'ORIGIN_freq']]
        
        # Add carrier columns in the right order
        for carrier in ['CARRIER_DH', 'CARRIER_DL', 'CARRIER_MQ', 'CARRIER_OH', 'CARRIER_RU', 'CARRIER_UA', 'CARRIER_US']:
            X_final[carrier] = carrier_dummies[carrier]
        
        return X_final[feature_order]


st.set_page_config(page_title="Logistic Regression Classifier", page_icon=":bar_chart:")
st.title("Flight Classification Predictor")
st.write("Enter the model inputs and get an instant classification result.")


@st.cache_resource
def load_pipeline():
    model_path = Path(__file__).resolve().parent / "pipeline.pkl"
    if not model_path.exists():
        raise FileNotFoundError("pipeline.pkl was not found in the app directory.")
    return joblib.load(model_path)


def extract_day_features(input_date: date) -> tuple:
    """Extract cyclical day of week features matching the model"""
    day_index = input_date.weekday()  # Monday=0 ... Sunday=6
    dow_sin = math.sin(2 * math.pi * day_index / 7)
    dow_cos = math.cos(2 * math.pi * day_index / 7)
    return dow_sin, dow_cos, day_index


def extract_time_features(input_time: time) -> tuple:
    """Extract cyclical time features matching the model"""
    time_in_hours = input_time.hour + input_time.minute / 60
    time_sin = math.sin(2 * math.pi * time_in_hours / 24)
    time_cos = math.cos(2 * math.pi * time_in_hours / 24)
    return time_sin, time_cos, time_in_hours


try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error(f"Unable to load model pipeline: {exc}")
    st.stop()


with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        weather = st.selectbox("Weather", options=["good", "bad"])
        carrier = st.selectbox("Carrier", options=["MQ", "RU", "UA", "OH", "DH", "DL", "US"])
        origin = st.selectbox("Origin Airport", options=["BWI", "IAD", "DCA"])
    
    with col2:
        flight_date = st.date_input("Date", value=date.today())
        scheduled_departure = st.time_input("Scheduled Depart Time", value=time(hour=12, minute=0))
        dest = st.selectbox("Destination Airport", options=["JFK", "LGA", "EWR"])

    submitted = st.form_submit_button("Get Classification")

if submitted:
    # Extract features from user inputs
    dow_sin, dow_cos, day_index = extract_day_features(flight_date)
    time_sin, time_cos, time_in_hours = extract_time_features(scheduled_departure)
    is_weekend = 1 if day_index >= 5 else 0
    
    # Encode weather as numeric
    weather_encoded = 0 if weather == "good" else 1
    
    # Create input dataframe with all required features
    engineered_features = {
        "WEATHER": weather_encoded,
        "CARRIER": carrier,
        "DEST": dest,
        "ORIGIN": origin,
        "DAY_WEEK": day_index,
        "DAY_OF_MONTH": flight_date.day,
        "DISTANCE": 200,
        "time_in_hours": time_in_hours,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "time_sin": time_sin,
        "time_cos": time_cos,
        "is_weekend": is_weekend,
    }

    input_df = pd.DataFrame([engineered_features])

    try:
        prediction = pipeline.predict(input_df)[0]
        prediction_label = "Delayed" if prediction == 1 else "On-time"
        st.success(f"Classification result: **{prediction_label}**")

        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(input_df)[0]
            classes = list(getattr(pipeline, "classes_", range(len(probs))))
            prob_df = pd.DataFrame({
                "class": ["On-time" if c == 0 else "Delayed" for c in classes],
                "probability": probs
            }).sort_values("probability", ascending=False)
            st.subheader("Class Probabilities")
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(
            "Prediction failed. This usually means the model expects different feature names or encodings. "
            f"Details: {exc}"
        )
