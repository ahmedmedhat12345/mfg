import openai
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Real Estate Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .prediction-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Models -----------------
@st.cache_resource
def load_models():
    models_dir = Path("models")
    if not models_dir.exists():
        st.error("Models directory not found")
        st.stop()

    required = ["regressor.pkl", "classifier.pkl", "kmeans.pkl", "scaler.pkl"]
    missing = [f for f in required if not (models_dir / f).exists()]
    if missing:
        st.error(f"Missing model files: {missing}")
        st.stop()

    try:
        regressor = joblib.load(models_dir / "regressor.pkl")
        classifier = joblib.load(models_dir / "classifier.pkl")
        kmeans = joblib.load(models_dir / "kmeans.pkl")
        scaler = joblib.load(models_dir / "scaler.pkl")
        return regressor, classifier, kmeans, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# ----------------- Feature Preparation -----------------
def prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode=None):
    features = {
        'bedrooms': float(bedrooms),
        'bathrooms': float(bathrooms),
        'sqft_living': float(sqft_living),
        'grade': float(grade) if grade else 7.0,
    }
    features['total_rooms'] = features['bedrooms'] + features['bathrooms']

    if zipcode:
        try:
            zipcode_str = str(zipcode).strip()
            features['zipcode_encoded'] = float(zipcode_str[-3:]) / 1000.0 if zipcode_str else 0.0
        except:
            features['zipcode_encoded'] = 0.0

    defaults = {
        'sqft_lot': 0.0, 'floors': 1.0, 'waterfront': 0.0, 'view': 0.0, 'condition': 5.0,
        'sqft_above': features['sqft_living'], 'sqft_basement': 0.0, 'yr_built': 2000.0,
        'yr_renovated': 0.0, 'lat': 47.5, 'long': -122.3, 'sqft_living15': features['sqft_living'], 'sqft_lot15': 0.0
    }
    for key, val in defaults.items():
        features.setdefault(key, val)
    return features

# ----------------- Predictions -----------------
def predict_cluster(features_dict, scaler, kmeans):
    cluster_features = ['bedrooms','bathrooms','sqft_living','grade','total_rooms']
    feature_vector = np.array([[features_dict.get(f,0) for f in cluster_features]])
    feature_vector_scaled = scaler.transform(feature_vector)
    cluster = kmeans.predict(feature_vector_scaled)[0]
    return cluster

def predict_price(features_dict, cluster_id, regressor):
    feature_names = ['bedrooms','bathrooms','sqft_living','grade','total_rooms','cluster']
    feature_vector = [features_dict.get(f, 0) if f != 'cluster' else float(cluster_id) for f in feature_names]
    feature_vector = np.array([feature_vector])
    predicted_price = max(0, regressor.predict(feature_vector)[0])
    return predicted_price

# ----------------- Built-in Explanation -----------------
def explain_prediction(features_dict, predicted_price, cluster_id):
    bedrooms = features_dict.get('bedrooms', 0)
    bathrooms = features_dict.get('bathrooms', 0)
    sqft_living = features_dict.get('sqft_living', 0)
    grade = features_dict.get('grade', 0)
    
    parts = [
        f"Predicted Value: ${predicted_price:,.2f}",
        f"Property: {bedrooms} bed, {bathrooms} bath, {sqft_living:,.0f} sqft"
    ]
    if grade: parts.append(f"Grade: {grade}")

    segment_names = {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}
    parts.append(f"Market Segment: {segment_names.get(cluster_id, f'Segment {cluster_id}')}")
    
    if sqft_living > 0:
        ppsf = predicted_price / sqft_living
        status = "below average" if ppsf < 200 else "average" if ppsf < 400 else "above average"
        parts.append(f"Price per sqft: ${ppsf:,.2f} ({status})")
    
    parts.append("Note: Consider location, condition, and market trends for final decision.")
    return "\n".join(parts)

# ----------------- LLM Explanation -----------------
def get_llm_explanation(features_dict, predicted_price):
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    prompt = (
        f"Explain this real estate prediction in plain English:\n\n"
        f"Features: {features_dict}\n"
        f"Predicted Price: ${predicted_price:,.2f}\n\n"
        f"Provide insights on market position, price per sqft, and investment considerations."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM explanation could not be generated: {e}"

# ----------------- Main App -----------------
def main():
    st.markdown('<div class="main-header">Real Estate Price Predictor</div>', unsafe_allow_html=True)
    
    with st.spinner("Loading models..."):
        regressor, classifier, kmeans, scaler = load_models()
    st.success("Models loaded successfully")
    
    with st.sidebar:
        st.header("Property Details")
        with st.form("prediction_form"):
            bedrooms = st.number_input("Bedrooms", 0, 10, 3, 1)
            bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, 0.5)
            sqft_living = st.number_input("Living Area (sqft)", 0, 20000, 2000, 100)
            grade = st.number_input("Property Grade", 1, 13, 7, 1)
            zipcode = st.text_input("Zipcode (Optional)", value="")
            submitted = st.form_submit_button("Predict Price", use_container_width=True)
    
    if submitted:
        features_dict = prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode)
        
        with st.spinner("Analyzing property..."):
            cluster_id = predict_cluster(features_dict, scaler, kmeans)
            predicted_price = predict_price(features_dict, cluster_id, regressor)
            explanation = explain_prediction(features_dict, predicted_price, cluster_id)
            llm_explanation = get_llm_explanation(features_dict, predicted_price)
        
        # Display results
        st.markdown("## Prediction Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Price", f"${predicted_price:,.0f}")
        col2.metric("Market Segment", {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}.get(cluster_id, f"Cluster {cluster_id}"))
        col3.metric("Price per SqFt", f"${predicted_price / sqft_living:,.2f}" if sqft_living > 0 else "N/A")
        
        st.markdown("---")
        st.markdown("## Prediction Explanation")
        st.markdown(f'<div class="prediction-box">{explanation}</div>', unsafe_allow_html=True)
        
        st.markdown("## LLM Interpretation")
        st.markdown(f'<div class="prediction-box">{llm_explanation}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        with st.expander("Additional Information"):
            st.info("Predictions are estimates based on property characteristics. Actual values may vary based on location, condition, and market trends.")
    else:
        st.info("Enter property details in the sidebar and click 'Predict Price'.")

if __name__ == "__main__":
    main()
