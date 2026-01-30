# Real Estate Price Predictor - Deployment Guide

This guide explains how to deploy the Real Estate Price Prediction app using Streamlit.

## 📁 Folder Structure

After setup, your project should have this structure:

```
real-estate-predictor/
│
├── models/                    # Model files (download from Kaggle)
│   ├── regressor.pkl         # XGBoost regression model
│   ├── classifier.pkl        # RandomForest classifier
│   ├── kmeans.pkl            # KMeans clustering model
│   └── scaler.pkl            # StandardScaler for features
│
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README_DEPLOYMENT.md      # This file
```

## 📥 Step 1: Download Models from Kaggle

The trained models are saved in your Kaggle notebook's output directory (`/kaggle/working/`).

### Option A: Download from Kaggle Notebook Output

1. **Open your Kaggle notebook** that trained the models
2. **Navigate to the "Output" tab** (or check `/kaggle/working/` directory)
3. **Download these files:**
   - `xgb_regressor.pkl` → Rename to `regressor.pkl`
   - `rf_classifier.pkl` → Rename to `classifier.pkl`
   - `kmeans_model.pkl` → Rename to `kmeans.pkl`
   - `scaler_cluster.pkl` → Rename to `scaler.pkl`

### Option B: Use Kaggle API

```bash
# Install Kaggle API (if not already installed)
pip install kaggle

# Download from your notebook output
kaggle kernels output [your-username]/[notebook-slug] -p models/
```

### Option C: Manual Download

1. In Kaggle notebook, go to **File → Download Output**
2. Extract the ZIP file
3. Copy the `.pkl` files to your local `models/` folder

## 📂 Step 2: Organize Model Files

1. **Create a `models/` folder** in your project directory:
   ```bash
   mkdir models
   ```

2. **Place all model files** in the `models/` folder:
   ```
   models/
   ├── regressor.pkl
   ├── classifier.pkl
   ├── kmeans.pkl
   └── scaler.pkl
   ```

3. **Verify file names match exactly:**
   - ✅ `regressor.pkl` (not `xgb_regressor.pkl`)
   - ✅ `classifier.pkl` (not `rf_classifier.pkl`)
   - ✅ `kmeans.pkl` (not `kmeans_model.pkl`)
   - ✅ `scaler.pkl` (not `scaler_cluster.pkl`)

## 🚀 Step 3: Local Setup & Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the App Locally

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`

### Verify It Works

1. Open the app in your browser
2. You should see: **"✅ Models loaded successfully!"**
3. Fill in property details in the sidebar
4. Click **"🔮 Predict Price"**
5. You should see predictions and explanations

## ☁️ Step 4: Deploy on Streamlit Cloud

### Prerequisites

- GitHub account
- Streamlit Cloud account (free at [streamlit.io](https://streamlit.io/cloud))

### Step-by-Step Deployment

#### 1. Push Code to GitHub

```bash
# Initialize git (if not already)
git init

# Add files
git add streamlit_app.py requirements.txt README_DEPLOYMENT.md

# Commit
git commit -m "Add Streamlit deployment app"

# Create GitHub repository and push
git remote add origin https://github.com/your-username/real-estate-predictor.git
git branch -M main
git push -u origin main
```

**Important:** Do NOT commit model files to GitHub (they're too large). Use `.gitignore`:

```bash
# Add to .gitignore
echo "models/*.pkl" >> .gitignore
```

#### 2. Upload Models to Streamlit Cloud

**Option A: Upload via Streamlit Cloud Secrets (Recommended for small models)**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. In app settings, use **"Secrets"** tab to upload model files (if under size limit)

**Option B: Use External Storage (Recommended for production)**

1. Upload models to cloud storage (AWS S3, Google Cloud Storage, etc.)
2. Modify `streamlit_app.py` to download models on startup
3. Add download logic in `load_models()` function

**Option C: Include Models in Repository (Not Recommended)**

- Only if models are small (< 100MB)
- Add models to repository
- Streamlit Cloud will include them

#### 3. Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**
2. **Click "New app"**
3. **Select your repository:**
   - Repository: `your-username/real-estate-predictor`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
4. **Click "Deploy"**
5. **Wait for deployment** (usually 1-2 minutes)
6. **Your app is live!** 🎉

### Streamlit Cloud Configuration

If you need to configure settings, create `config.toml`:

```toml
[server]
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false
```

## 🔍 Verify Model Compatibility

If you want to check what features your models expect, you can run this Python script:

```python
import joblib
import numpy as np

# Load models
regressor = joblib.load("models/regressor.pkl")
scaler = joblib.load("models/scaler.pkl")

# Check feature names (if available)
if hasattr(regressor, 'feature_names_in_'):
    print("Regressor features:", regressor.feature_names_in_)
if hasattr(scaler, 'feature_names_in_'):
    print("Scaler features:", scaler.feature_names_in_)

# For XGBoost models
try:
    if hasattr(regressor, 'get_booster'):
        print("XGBoost features:", regressor.get_booster().feature_names)
except:
    pass
```

If your models use different feature names, you may need to adjust `prepare_features()` and `predict_price()` functions in `streamlit_app.py`.

## 🔧 Troubleshooting

### Issue: "Model file not found"

**Solution:**
- Check that `models/` folder exists
- Verify all 4 `.pkl` files are present
- Ensure file names match exactly (case-sensitive)

### Issue: "Error loading models"

**Solution:**
- Ensure models were saved with `joblib` (not `pickle`)
- Check Python version compatibility
- Verify scikit-learn and xgboost versions match training environment

### Issue: "App crashes on Streamlit Cloud"

**Solution:**
- Check Streamlit Cloud logs for error messages
- Ensure `requirements.txt` includes all dependencies
- Verify model files are accessible (if using external storage)

### Issue: "Predictions seem incorrect"

**Solution:**
- Verify feature names match training data
- Check that input ranges are similar to training data
- Ensure models were trained on similar data distribution

## 📊 Model Information

### Model Details

- **Regressor:** XGBoost (predicts exact price)
- **Classifier:** RandomForest (categorizes price range)
- **Clustering:** KMeans (identifies market segment)
- **Scaler:** StandardScaler (normalizes features)

### Expected Input Features

- `bedrooms`: Integer (0-10)
- `bathrooms`: Float (0.0-10.0)
- `sqft_living`: Integer (living area in square feet)
- `grade`: Integer (1-13, property quality grade)
- `zipcode`: String (optional, for location context)

### Output

- **Predicted Price:** Dollar amount
- **Market Segment:** Budget / Mid-Market / Premium / Luxury
- **Price per SqFt:** Calculated metric
- **Natural Language Explanation:** Detailed insights

## 🎯 Next Steps

### Enhancements

1. **Add more features:** Year built, lot size, condition, etc.
2. **Location features:** Use zipcode for location-based pricing
3. **Model retraining:** Periodically retrain with new data
4. **API integration:** Connect to real estate APIs for live data
5. **Visualizations:** Add charts for price trends, comparisons

### Production Considerations

- **Model versioning:** Track which model version is deployed
- **Monitoring:** Log predictions and monitor model performance
- **A/B testing:** Test new models before full deployment
- **Error handling:** Add robust error handling for edge cases
- **Caching:** Optimize model loading and prediction speed

## 📝 Notes

- Models are trained on historical data and may not reflect current market conditions
- Predictions are estimates and should not replace professional appraisals
- Model performance depends on data quality and feature engineering
- Regular model updates improve accuracy over time

## 🆘 Support

If you encounter issues:

1. Check this README for common solutions
2. Review Streamlit Cloud logs
3. Verify model files are correct
4. Test locally before deploying

---

**Happy Deploying! 🚀**

