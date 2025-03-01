import gdown
import joblib
import streamlit as st
import os
from main import collect_inputs

# ✅ Your Google Drive file ID
MODEL_URL = "https://drive.google.com/uc?id=1W9TTRRSrMwy9U3fVYc83nQETZ9B9Mp68"

@st.cache_resource
def load_model():
    model_path = "random_forest_model.pkl"
    if not os.path.exists(model_path):
        st.info("Downloading model... ⏳")
        gdown.download(MODEL_URL, model_path, quiet=False)
    return joblib.load(model_path)

model = load_model()

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "input"

# **Page 1: Collect Inputs**
if st.session_state.page == "input":
    st.title("✈️ Flight Price Predictor")

    # Get user inputs
    user_inputs = collect_inputs()

    if st.button("Search Flights"):
        st.session_state.page = "results"
        st.session_state.user_inputs = user_inputs
        st.rerun()

# **Page 2: Show Predictions**
elif st.session_state.page == "results":
    st.title("🔮 Predicted Flight Price")

    # Retrieve stored user inputs
    user_inputs = st.session_state.user_inputs

    # Make prediction
    predicted_price = model.predict(user_inputs)[0]

    # Display result
    st.success(f"💰 Estimated Flight Price: **${predicted_price:.2f}**")

    # Go back button
    if st.button("🔄 Try Another Search"):
        st.session_state.page = "input"
        st.rerun()