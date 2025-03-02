import os
import joblib
import streamlit as st
import traceback
from main import collect_inputs

# Define model file path
MODEL_PATH = "random_forest_model.pkl"

@st.cache_resource
def load_model():
    """Load the trained Random Forest model from the local file."""
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

    # # Debug: Show input feature names to confirm correct format
    # st.write("✅ Input Features:", user_inputs.columns.tolist())

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
