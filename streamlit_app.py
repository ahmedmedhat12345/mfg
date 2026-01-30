"""
Real Estate Price Prediction App
Loads trained models and provides price predictions.
"""

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models_dir = Path("models")
    
    if not models_dir.exists():
        st.error("Models directory not found")
        st.error(f"Expected: {models_dir.absolute()}")
        st.info("Create a 'models/' folder and place model files there.")
        st.stop()
    
    existing_files = list(models_dir.glob("*.pkl"))
    required = ["regressor.pkl", "classifier.pkl", "kmeans.pkl", "scaler.pkl"]
    
    missing = [f for f in required if not (models_dir / f).exists()]
    
    if missing:
        st.error("Missing model files")
        for f in required:
            status = "Found" if (models_dir / f).exists() else "Missing"
            st.text(f"{status}: {f}")
        
        if existing_files:
            st.warning("Found these .pkl files:")
            for f in existing_files:
                st.code(f.name)
            st.info("Rename files: xgb_regressor.pkl -> regressor.pkl, kmeans_model.pkl -> kmeans.pkl, etc.")
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


def explain_prediction(features_dict, predicted_price, cluster_id):
    bedrooms = features_dict.get('bedrooms', 0)
    bathrooms = features_dict.get('bathrooms', 0)
    sqft_living = features_dict.get('sqft_living', 0)
    grade = features_dict.get('grade', 0)
    
    parts = []
    parts.append(f"Predicted Value: ${predicted_price:,.2f}")
    parts.append(f"\nProperty: {bedrooms} bed, {bathrooms} bath, {sqft_living:,.0f} sqft")
    if grade:
        parts.append(f"Grade: {grade}")
    
    segment_names = {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}
    segment = segment_names.get(cluster_id, f"Segment {cluster_id}")
    parts.append(f"\nMarket Segment: {segment}")
    
    if sqft_living > 0:
        ppsf = predicted_price / sqft_living
        parts.append(f"Price per sqft: ${ppsf:,.2f}")
        
        if ppsf < 200:
            status = "below average"
        elif ppsf < 400:
            status = "average"
        else:
            status = "above average"
        parts.append(f"Market position: {status}")
    
    parts.append("\nNote: Consider location, condition, and market trends for final decision.")
    
    return "\n".join(parts)


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
            if zipcode_str:
                features['zipcode_encoded'] = float(zipcode_str[-3:]) / 1000.0
            else:
                features['zipcode_encoded'] = 0.0
        except:
            features['zipcode_encoded'] = 0.0
    
    defaults = {
        'sqft_lot': 0.0,
        'floors': 1.0,
        'waterfront': 0.0,
        'view': 0.0,
        'condition': 5.0,
        'sqft_above': features['sqft_living'],
        'sqft_basement': 0.0,
        'yr_built': 2000.0,
        'yr_renovated': 0.0,
        'lat': 47.5,
        'long': -122.3,
        'sqft_living15': features['sqft_living'],
        'sqft_lot15': 0.0,
    }
    
    for key, val in defaults.items():
        if key not in features:
            features[key] = val
    
    return features


def predict_cluster(features_dict, scaler, kmeans):
    priority = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms']
    
    try:
        if hasattr(scaler, 'feature_names_in_'):
            cluster_features = list(scaler.feature_names_in_)
        else:
            cluster_features = [f for f in priority if f in features_dict]
    except:
        cluster_features = [f for f in priority if f in features_dict]
    
    if not cluster_features:
        cluster_features = ['bedrooms', 'bathrooms', 'sqft_living']
    
    feature_vector = np.array([[features_dict.get(f, 0) for f in cluster_features]])
    feature_vector_scaled = scaler.transform(feature_vector)
    cluster = kmeans.predict(feature_vector_scaled)[0]
    
    return cluster


def predict_price(features_dict, cluster_id, regressor):
    try:
        if hasattr(regressor, 'feature_names_in_'):
            feature_names = list(regressor.feature_names_in_)
        elif hasattr(regressor, 'get_booster'):
            feature_names = regressor.get_booster().feature_names
            if feature_names is None:
                raise AttributeError
        else:
            raise AttributeError
    except (AttributeError, TypeError):
        feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 'total_rooms', 'cluster']
        possible = ['sqft_lot', 'floors', 'waterfront', 'view', 'condition', 
                   'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                   'lat', 'long', 'sqft_living15', 'sqft_lot15']
        feature_names.extend([f for f in possible if f in features_dict])
    
    feature_vector = []
    for name in feature_names:
        if name == 'cluster':
            feature_vector.append(float(cluster_id))
        elif name in features_dict:
            feature_vector.append(float(features_dict[name]))
        else:
            feature_vector.append(0.0)
    
    feature_vector = np.array([feature_vector])
    predicted_price = regressor.predict(feature_vector)[0]
    predicted_price = max(0, predicted_price)
    
    return predicted_price


def main():
    st.markdown('<div class="main-header">Real Estate Price Predictor</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    with st.spinner("Loading models..."):
        regressor, classifier, kmeans, scaler = load_models()
    
    st.success("Models loaded successfully")
    
    with st.sidebar:
        st.header("Property Details")
        
        with st.form("prediction_form"):
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            sqft_living = st.number_input("Living Area (sqft)", min_value=0, max_value=20000, value=2000, step=100)
            grade = st.number_input("Property Grade", min_value=1, max_value=13, value=7, step=1)
            zipcode = st.text_input("Zipcode (Optional)", value="")
            
            submitted = st.form_submit_button("Predict Price", use_container_width=True, type="primary")
    
    if submitted:
        features_dict = prepare_features(bedrooms, bathrooms, sqft_living, grade, zipcode if zipcode else None)
        
        with st.spinner("Analyzing property..."):
            cluster_id = predict_cluster(features_dict, scaler, kmeans)
            predicted_price = predict_price(features_dict, cluster_id, regressor)
            explanation = explain_prediction(features_dict, predicted_price, cluster_id)
        
        st.markdown("## Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Price", f"${predicted_price:,.0f}")
        
        with col2:
            segment_names = {0: "Budget", 1: "Mid-Market", 2: "Premium", 3: "Luxury"}
            st.metric("Market Segment", segment_names.get(cluster_id, f"Cluster {cluster_id}"))
        
        with col3:
            if sqft_living > 0:
                ppsf = predicted_price / sqft_living
                st.metric("Price per SqFt", f"${ppsf:,.2f}")
        
        st.markdown("---")
        st.markdown("## Prediction Explanation")
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(explanation)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        with st.expander("Additional Information"):
            st.info("""
            Predictions are estimates based on property characteristics.
            Actual values may vary based on location, condition, and market conditions.
            Consult with real estate professionals for accurate valuations.
            """)
    
    else:
        st.info("Enter property details in the sidebar and click 'Predict Price'.")


if __name__ == "__main__":
    main()
