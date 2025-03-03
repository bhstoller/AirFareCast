import os
import gdown
import joblib
import streamlit as st
import traceback
from main import collect_inputs

# ✅ Google Drive file ID for the trained Random Forest model
MODEL_URL = "https://drive.google.com/uc?id=1W9TTRRSrMwy9U3fVYc83nQETZ9B9Mp68"

# Define model file path
MODEL_PATH = "random_forest_model.pkl"

@st.cache_resource
def load_model():
    """Download and load the trained Random Forest model."""
    # ✅ Check if the model exists
    if not os.path.exists(MODEL_PATH):
        st.info("🔽 Downloading model... This may take a few seconds.")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # ✅ Ensure model was downloaded successfully
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ Model download failed. Check Google Drive link.")

    return joblib.load(MODEL_PATH)

# Load model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.text(traceback.format_exc())  # Show full traceback
    st.stop()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"

# **Page 1: Collect Inputs**
if st.session_state.page == "input":
    st.title("✈️ Flight Price Predictor")

    # Get user inputs with error handling
    try:
        user_inputs = collect_inputs()
    except Exception as e:
        st.error("❌ Error collecting inputs")
        st.text(traceback.format_exc())  # Show full traceback
        st.stop()

    if st.button("Search Flights"):
        st.session_state.page = "results"
        st.session_state.user_inputs = user_inputs
        st.rerun()

# **Page 2: Show Predictions**
elif st.session_state.page == "results":
    st.title("🔮 Predicted Flight Price")

    # Retrieve stored user inputs
    user_inputs = st.session_state.user_inputs

    # Debug: Show input feature names to confirm correct format
    st.write("✅ Input Features:", user_inputs.columns.tolist())

    # Try making prediction
    try:
        predicted_price = model.predict(user_inputs)[0]
        st.success(f"💰 Estimated Flight Price: **${predicted_price:.2f}**")
    except Exception as e:
        st.error("❌ Error during prediction")
        st.text(traceback.format_exc())  # Show full error traceback
        st.stop()

    # Button to go back and try another search
    if st.button("🔄 Try Another Search"):
        st.session_state.page = "input"
        st.rerun()
