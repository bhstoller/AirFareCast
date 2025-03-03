import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import json
import traceback
import matplotlib.pyplot as plt

# Load feature order
feature_order = joblib.load("feature_order.pkl")

# Load travel data
with open("travel_data.json", "r") as f:
    TRAVEL_DATA_JSON = json.load(f)

# Convert string keys back to tuple keys
TRAVEL_DATA = {
    tuple(key.split("_")): value for key, value in TRAVEL_DATA_JSON.items()
}

# Extract valid origins and destinations
VALID_ORIGINS = list(set(origin for origin, _ in TRAVEL_DATA.keys()))
VALID_DESTINATIONS = {
    origin: [dest for (orig, dest) in TRAVEL_DATA.keys() if orig == origin]
    for origin in VALID_ORIGINS
}

# Collect all airports
AIRPORTS = sorted(set(VALID_ORIGINS + [dest for sublist in VALID_DESTINATIONS.values() for dest in sublist]))

# Dictionary of holiday dates
with open("holiday_dates.json", "r") as f:
    HOLIDAY_DATES = json.load(f)

# Convert holidays to datetime format
HOLIDAY_DATES = {name: datetime.datetime.strptime(date, "%Y-%m-%d").date() for name, date in HOLIDAY_DATES.items()}

# Function to determine if a date is a holiday or near a holiday
def add_holiday_features(departure_date):
    is_holiday = int(departure_date in HOLIDAY_DATES.values())
    near_holiday = int(any(abs((departure_date - holiday).days) <= 3 for holiday in HOLIDAY_DATES.values()))
    return is_holiday, near_holiday

# Function to extract features from user inputs
def collect_inputs():
    departure = st.selectbox("Departure Airport", VALID_ORIGINS)
    valid_arrivals = VALID_DESTINATIONS.get(departure, [])
    arrival = st.selectbox("Arrival Airport", valid_arrivals)

    if not arrival:
        st.error("No valid destinations for this origin. Please select a different departure airport.")
        return None
    
    is_refundable = int(st.checkbox("Refundable Ticket"))
    is_nonstop = int(st.checkbox("Nonstop Flight"))
    
    departure_date = st.date_input("Departure Date", min_value=datetime.date.today())
    days_to_departure = (departure_date - datetime.date.today()).days
    departure_day_of_week = departure_date.weekday()
    is_weekend = 1 if departure_day_of_week in [5, 6] else 0

    departure_time = st.text_input("Enter Departure Time (HH:MM)", "12:00")
    departure_hour, departure_min = map(int, departure_time.split(":"))
    departure_time_float = departure_hour + (departure_min / 60)

    is_holiday, near_holiday = add_holiday_features(departure_date)

    travel_info = TRAVEL_DATA.get((departure, arrival), {"distance": 0, "duration": 0})
    travel_distance = travel_info["distance"]
    travel_duration = travel_info["duration"]

    cabin_class = st.selectbox("Cabin Class", ["Economy", "Premium Economy", "Business", "Basic Economy"])
    cabin_class_mapping = {"Basic Economy": 1, "Economy": 0, "Premium Economy": 0, "Business": 0}

    seats_remaining = st.slider("Seats Remaining", min_value=1, max_value=10, value=5)
    binned_seats_1 = 1 if seats_remaining <= 2 else 0
    binned_seats_2 = 1 if 3 <= seats_remaining <= 5 else 0

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

    for airport in AIRPORTS:
        user_inputs[f"startingAirport_{airport}"] = 1 if departure == airport else 0
        user_inputs[f"destinationAirport_{airport}"] = 1 if arrival == airport else 0

    user_inputs_df = pd.DataFrame([user_inputs])

    for col in feature_order:
        if col not in user_inputs_df.columns:
            user_inputs_df[col] = 0
    user_inputs_df = user_inputs_df[feature_order]

    return user_inputs_df

# Function to predict price forecast
def predict_price_forecast(user_inputs_df, model):
    forecast_data = []
    today_days_to_departure = int(user_inputs_df["daysToDeparture"][0])

    for days_out in range(today_days_to_departure, -1, -1):
        modified_input = user_inputs_df.copy()
        modified_input["daysToDeparture"] = days_out
        predicted_price = model.predict(modified_input)[0]
        forecast_data.append({"Days to Departure": days_out, "Predicted Price ($)": predicted_price})

    df_forecast = pd.DataFrame(forecast_data)
    cheapest_day = df_forecast.loc[df_forecast["Predicted Price ($)"].idxmin(), "Days to Departure"]
    
    # Store raw price values separately for plotting
    df_forecast["Raw Price"] = df_forecast["Predicted Price ($)"]
    
    # Apply Markdown formatting for bolding the lowest price row and its corresponding day
    df_forecast["Formatted Days"] = df_forecast.apply(
        lambda row: f"**{row['Days to Departure']}**" if row["Days to Departure"] == cheapest_day else f"{row['Days to Departure']}",
        axis=1
    )
    df_forecast["Formatted Price"] = df_forecast.apply(
        lambda row: f"**${row['Predicted Price ($)']:.2f}**" if row["Days to Departure"] == cheapest_day else f"${row['Predicted Price ($)']:.2f}",
        axis=1
    )
    
    return df_forecast.reset_index(drop=True), cheapest_day

# Function to display results
def display_results(model):
    st.write("# ✈️ Flight Price Predictor")

    user_inputs = st.session_state.user_inputs

    try:
        predicted_price = model.predict(user_inputs)[0]
        price_range = 50

        st.write(f"### 💰 **Expected Price:** ${predicted_price:.2f} ± ${price_range}")

        st.write("---")

        df_forecast, cheapest_day = predict_price_forecast(user_inputs, model)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("## 📉 Best Time to Book ↔ Price Forecast")
            st.table(df_forecast[["Formatted Days", "Formatted Price"]].rename(columns={"Formatted Days": "Days to Departure", "Formatted Price": "Predicted Price ($)"}))

        with col2:
            st.write("## 📊 Price Forecast")
            csfont = {'fontname':'sans serif'}
            fig, ax = plt.subplots(figsize= (8,4))
            ax.plot(df_forecast["Days to Departure"], df_forecast["Raw Price"], marker='o', linestyle='-')
            ax.set_xlabel("Days to Departure", **csfont)
            ax.set_ylabel("Predicted Price ($)", **csfont)
            ax.set_title("Price Forecast", **csfont, fontweight= 'bold', fontsize= 14)
            ax.set_xlim(df_forecast["Days to Departure"].min(), df_forecast["Days to Departure"].max())
            ax.set_ylim(df_forecast["Raw Price"].min() * 0.9, df_forecast["Raw Price"].max() * 1.1)  # Add some margin
            
            st.pyplot(fig)


        if cheapest_day == user_inputs["daysToDeparture"][0]:
            st.write(f"### 📌 **Recommendation: Buy Now!** Prices are expected to **increase** as departure nears.")
        else:
            st.write(f"### 📌 **Recommendation: Wait!** The price is expected to be **lowest on {cheapest_day} days before departure**.")
        st.write("---")

    except Exception as e:
        st.error("❌ Error during prediction")
        st.text(traceback.format_exc())
        st.stop()

    if st.button("🔄 Search Again"):
        st.session_state.page = "input"
        st.rerun()


