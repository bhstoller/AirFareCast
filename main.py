import streamlit as st
import pandas as pd
import datetime
import joblib

# Load feature order
feature_order = joblib.load("feature_order.pkl")

# List of airports based on the trained model
AIRPORTS = ["BOS", "CLT", "DEN", "DFW", "DTW", "EWR", "IAD", "JFK", "LAX", "LGA",
            "MIA", "OAK", "ORD", "PHL", "SFO"]

# Dictionary of holiday dates
HOLIDAY_DATES = {
    "Easter Sunday": "2022-04-17",
    "Independence Day": "2022-07-04",
    "Mother's Day": "2022-05-08",
    "Labor Day": "2022-09-05",
    "Columbus Day": "2022-10-10",
    "Veterans Day": "2022-11-11"
}

# Convert holidays to datetime format
HOLIDAY_DATES = {name: datetime.datetime.strptime(date, "%Y-%m-%d").date() for name, date in HOLIDAY_DATES.items()}

# Function to determine if a date is a holiday or near a holiday
def add_holiday_features(departure_date):
    is_holiday = int(departure_date in HOLIDAY_DATES.values())
    near_holiday = int(any(abs((departure_date - holiday).days) <= 3 for holiday in HOLIDAY_DATES.values()))
    return is_holiday, near_holiday

# Function to extract features from user inputs
def collect_inputs():
    departure = st.selectbox("Departure Airport", AIRPORTS)
    arrival = st.selectbox("Arrival Airport", [airport for airport in AIRPORTS if airport != departure])
    is_refundable = int(st.checkbox("Refundable Ticket"))
    is_nonstop = int(st.checkbox("Nonstop Flight"))
    
    departure_date = st.date_input("Departure Date", min_value=datetime.date.today())
    days_to_departure = (departure_date - datetime.date.today()).days
    departure_day_of_week = departure_date.weekday()
    is_weekend = 1 if departure_day_of_week in [5, 6] else 0  # Sat/Sun

    departure_time = st.text_input("Enter Departure Time (HH:MM)", "12:00")
    departure_hour, departure_min = map(int, departure_time.split(":"))
    departure_time_float = departure_hour + (departure_min / 60)

    # Calculate holiday features
    is_holiday, near_holiday = add_holiday_features(departure_date)

    # Placeholder travel data (replace with real API calculations if available)
    travel_duration = 360  # Example value
    travel_distance = 1000  # Example value

    # Cabin Class Encoding
    cabin_class = st.selectbox("Cabin Class", ["Basic Economy", "Main Cabin", "Business"])
    cabin_class_mapping = {"Basic Economy": 1, "Main Cabin": 0, "Business": 0}  # One-hot encoding

    # Seats Remaining Binning
    seats_remaining = st.slider("Seats Remaining", min_value=0, max_value=10, value=5)
    binned_seats_1 = 1 if seats_remaining <= 2 else 0
    binned_seats_2 = 1 if 3 <= seats_remaining <= 5 else 0

    # Initialize input dictionary
    user_inputs = {
        'travelDuration': travel_duration,
        'isRefundable': is_refundable,
        'isNonStop': is_nonstop,
        'seatsRemaining': seats_remaining,
        'travelDistance': travel_distance,
        'departureTimeHour': departure_hour,
        'departureTimeFloat': departure_time_float,
        'daysToDeparture': days_to_departure,
        'departureDayOfWeek': departure_day_of_week,
        'isWeekend': is_weekend,
        'isHoliday': is_holiday,
        'nearHoliday': near_holiday,
        'cabinClass_basic economy': cabin_class_mapping[cabin_class],
        'binnedSeatsRemaining_1': binned_seats_1,
        'binnedSeatsRemaining_2': binned_seats_2
    }

    # One-Hot Encoding for Departure and Arrival Airports
    for airport in AIRPORTS:
        user_inputs[f"startingAirport_{airport}"] = 1 if departure == airport else 0
        user_inputs[f"destinationAirport_{airport}"] = 1 if arrival == airport else 0

    # Convert dictionary to DataFrame
    user_inputs_df = pd.DataFrame([user_inputs])

    # Ensure feature order matches training
    for col in feature_order:
        if col not in user_inputs_df.columns:
            user_inputs_df[col] = 0  # Fill missing columns with 0
    user_inputs_df = user_inputs_df[feature_order]  # Reorder columns

    return user_inputs_df
