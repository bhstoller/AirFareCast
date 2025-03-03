# model.py

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load pre-trained model (cache for faster performance)
@st.cache_resource
def load_model():
    with open("random_forest_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Full travel data dictionary
TRAVEL_DATA = {
    ("ATL", "BOS"): {"travelDuration": 360.5, "travelDistance": 999.33},
    ("ATL", "CLT"): {"travelDuration": 257.4, "travelDistance": 673.14},
    ("ATL", "DEN"): {"travelDuration": 394.03, "travelDistance": 1525.21},
    ("ATL", "DFW"): {"travelDuration": 271.96, "travelDistance": 912.04},
    ("ATL", "DTW"): {"travelDuration": 362.05, "travelDistance": 1024.93},
    ("ATL", "EWR"): {"travelDuration": 278.16, "travelDistance": 846.68},
    ("ATL", "IAD"): {"travelDuration": 237.39, "travelDistance": 675.6},
    ("ATL", "JFK"): {"travelDuration": 321.96, "travelDistance": 927.8},
    ("ATL", "LAX"): {"travelDuration": 492.35, "travelDistance": 2086.36},
    ("ATL", "LGA"): {"travelDuration": 274.04, "travelDistance": 881.87},
    ("ATL", "MIA"): {"travelDuration": 287.59, "travelDistance": 1096.5},
    ("ATL", "OAK"): {"travelDuration": 724.6, "travelDistance": 2415.59},
    ("ATL", "ORD"): {"travelDuration": 361.08, "travelDistance": 988.9},
    ("ATL", "PHL"): {"travelDuration": 291.55, "travelDistance": 865.21},
    ("ATL", "SFO"): {"travelDuration": 592.08, "travelDistance": 2509.35},
    ("BOS", "ATL"): {"travelDuration": 361.66, "travelDistance": 1028.47},
    ("BOS", "CLT"): {"travelDuration": 319.36, "travelDistance": 784.04},
    ("BOS", "DEN"): {"travelDuration": 498.77, "travelDistance": 1878.43},
    ("BOS", "DFW"): {"travelDuration": 449.28, "travelDistance": 1638.1},
    ("BOS", "DTW"): {"travelDuration": 374.06, "travelDistance": 781.52},
    ("BOS", "EWR"): {"travelDuration": 280.96, "travelDistance": 737.03},
    ("BOS", "IAD"): {"travelDuration": 306.95, "travelDistance": 672.08},
    ("BOS", "JFK"): {"travelDuration": 136.6, "travelDistance": 249.59},
    ("BOS", "LAX"): {"travelDuration": 547.13, "travelDistance": 2701.6},
    ("BOS", "LGA"): {"travelDuration": 204.53, "travelDistance": 483.54},
    ("BOS", "MIA"): {"travelDuration": 364.97, "travelDistance": 1310.48},
    ("BOS", "OAK"): {"travelDuration": 771.67, "travelDistance": 3005.73},
    ("BOS", "ORD"): {"travelDuration": 279.49, "travelDistance": 906.2},
    ("BOS", "PHL"): {"travelDuration": 257.38, "travelDistance": 310.78},
    ("BOS", "SFO"): {"travelDuration": 575.7, "travelDistance": 2793.93}
    # Add remaining routes here...
}

# Function to predict flight price
def predict_price(departure, arrival, is_refundable, is_nonstop, 
                  departure_date, departure_time, seats_remaining, cabin_class):
    
    # Extract known travel duration and distance
    travel_info = TRAVEL_DATA.get((departure, arrival), {"travelDuration": 0, "travelDistance": 0})

    # Convert departure date into useful features
    departure_datetime = pd.to_datetime(departure_date + " " + departure_time)
    days_to_departure = (departure_datetime - pd.Timestamp.now()).days
    departure_hour = departure_datetime.hour
    departure_day_of_week = departure_datetime.weekday()
    is_weekend = 1 if departure_day_of_week in [5, 6] else 0

    # Prepare input data
    input_data = {
        "travelDuration": travel_info["travelDuration"],
        "travelDistance": travel_info["travelDistance"],
        "isRefundable": int(is_refundable),
        "isNonStop": int(is_nonstop),
        "departureTimeHour": departure_hour,
        "departureDayOfWeek": departure_day_of_week,
        "daysToDeparture": days_to_departure,
        "isWeekend": is_weekend,
        "seatsRemaining": seats_remaining,
        "cabinClass_basic economy": 1 if cabin_class == "Economy" else 0,
    }

    # Convert to DataFrame and ensure all required features are present
    feature_df = pd.DataFrame([input_data])

    # Predict price
    predicted_price = model.predict(feature_df)[0]
    return predicted_price
