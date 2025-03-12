import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from datetime import datetime


def adjust_cabin_class(val):
    mapping = {
        "coach": "economy",
        "premium coach": "premium economy",
        "business": "first"
    }
    return mapping.get(val, val)


def clean_encoded_value(val):
    """Convert value to string and, if it contains '||', return the part before it."""
    val_str = str(val)
    if "||" in val_str:
        return val_str.split("||")[0]
    return val_str


def encode_value(val, encoder):
    """
    Given a value and a LabelEncoder, clean the value and then
    transform it using the encoder's mapping (after cleaning the encoder classes).
    This returns the original code that the model expects.
    """
    cleaned_val = clean_encoded_value(val)
    mapping = {clean_encoded_value(x): x for x in encoder.classes_}
    if cleaned_val not in mapping:
        st.error(f"Value '{cleaned_val}' not found in label encoder classes.")
        return None
    orig_val = mapping[cleaned_val]
    return encoder.transform([orig_val])[0]


@st.cache_data
def load_csv_data(path):
    return pd.read_csv(path, parse_dates=['searchDate', 'flightDate'])[:25000]


@st.cache_resource
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


@st.cache_resource
def load_label_encoders(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def autoregressive_forecast(base_row, model, model_features, n_forecast=7, label_encoders=None):
    """
    Given a base_row (from the CSV) with historical lagged prices,
    predict the next n_forecast days autoregressively.
    If label_encoders are provided, encode categorical features using cleaned values.

    Returns:
      historical: list of 7 historical prices (oldest to most recent)
      predictions: list of n_forecast forecasted prices.
    """
    historical = [base_row[f'price_t_minus_{i}'] for i in range(7, 0, -1)]
    current_lags = np.array([base_row[f'price_t_minus_{i}'] for i in range(1, 8)])
    predictions = []
    current_row = base_row.copy()

    if label_encoders is not None:
        for col in ['startingAirport', 'destinationAirport', 'airlineCode', 'cabinClass']:
            encoded_val = encode_value(current_row[col], label_encoders[col])
            if encoded_val is None:
                return historical, []
            current_row[col] = encoded_val

    for _ in range(n_forecast):
        for j, lag_val in enumerate(current_lags, start=1):
            current_row[f'price_t_minus_{j}'] = lag_val
        features = current_row[model_features].values.reshape(1, -1)
        pred = model.predict(features)[0]
        pred = math.floor(pred / 10) * 10  # floor prediction to 10s place
        predictions.append(pred)
        current_lags = np.insert(current_lags[:-1], 0, pred)
        current_row['totalFare'] = pred
    return historical, predictions


data_path = "data/sample_data_lagged.csv"
model_path = "submission/xg_boost_model.pkl"
encoders_path = "label_encoders.pkl"

df = load_csv_data(data_path)
model = load_model(model_path)
label_encoders = load_label_encoders(encoders_path)

# Convert numeric codes to strings for startingAirport and destinationAirport
df['startingAirport'] = df['startingAirport'].apply(
    lambda x: label_encoders['startingAirport'].inverse_transform([int(x)])[0] if pd.notnull(x) else x)
df['destinationAirport'] = df['destinationAirport'].apply(
    lambda x: label_encoders['destinationAirport'].inverse_transform([int(x)])[0] if pd.notnull(x) else x)

# For cabinClass and airlineCode, store the original ("model") values
# then create cleaned display versions since cannot be inverted
df['cabinClass_model'] = df['cabinClass'].apply(
    lambda x: label_encoders['cabinClass'].inverse_transform([int(x)])[0] if pd.notnull(x) else x)
df['cabinClass'] = df['cabinClass_model'].apply(clean_encoded_value)
df['cabinClass'] = df['cabinClass'].apply(adjust_cabin_class)

df['airlineCode_model'] = df['airlineCode'].apply(
    lambda x: label_encoders['airlineCode'].inverse_transform([int(x)])[0] if pd.notnull(x) else x)
df['airlineCode'] = df['airlineCode_model'].apply(clean_encoded_value)

model_features = [
    'startingAirport', 'destinationAirport', 'travelDuration', 'isRefundable', 'isNonStop',
    'totalFare', 'seatsRemaining', 'airlineCode', 'cabinClass', 'travelDistance',
    'departureTimeHour', 'daysToDeparture', 'departureDayOfWeek', 'isWeekend', 'isHoliday', 'nearHoliday',
    'price_t_minus_1', 'price_t_minus_2', 'price_t_minus_3', 'price_t_minus_4', 'price_t_minus_5',
    'price_t_minus_6', 'price_t_minus_7'
]

# airline mapping for display
airline_code_dictionary = {
    'UA': 'united',
    'AA': 'american',
    'DL': 'delta',
    'WN': 'southwest',
    'F9': 'frontier',
    'NK': 'spirit',
    'B6': 'jetblue',
    'AS': 'alaska'
}

airline_mapping = {
    'united': ('United Airlines', 'https://www.united.com'),
    'american': ('American Airlines', 'https://www.aa.com'),
    'delta': ('Delta Air Lines', 'https://www.delta.com'),
    'southwest': ('Southwest Airlines', 'https://www.southwest.com'),
    'frontier': ('Frontier Airlines', 'https://www.flyfrontier.com'),
    'spirit': ('Spirit Airlines', 'https://www.spirit.com'),
    'jetblue': ('JetBlue', 'https://www.jetblue.com'),
    'alaska': ('Alaska Airlines', 'https://www.alaskaair.com')
}

# dropdown options
starting_airport_options = sorted(set(df['startingAirport'].unique()))
destination_airport_options = sorted(set(df['destinationAirport'].unique()))
cabin_class_options = sorted(set(df['cabinClass'].unique()))

# display valid flight combinations for debugging purposes
# unique_itineraries = df
# unique_itineraries = unique_itineraries[unique_itineraries['startingAirport'] == "ORD"]
# unique_itineraries = unique_itineraries[unique_itineraries['destinationAirport'] == "SFO"]
# sample_itineraries = unique_itineraries #.head(10).copy()
# sample_itineraries['flightDate'] = sample_itineraries['flightDate'].dt.date
# st.markdown("### 10 Valid Flight Combinations Available")
# st.dataframe(sample_itineraries)

st.title("AirFareCast: Flight Price Forecasting")
st.write("Enter when you want to fly, and we'll tell you when to book.")

demo_today = datetime(2022, 5, 1)
st.write(f"Today's Date: **{demo_today.strftime('%Y-%m-%d')}**")

with st.form("itinerary_form"):
    flight_date = st.date_input("Flight Date", value=demo_today)
    starting_airport = st.selectbox("Starting Airport", options=starting_airport_options)
    destination_airport = st.selectbox("Destination Airport", options=destination_airport_options)
    is_nonstop = st.checkbox("Nonstop Flight", value=False)
    cabin_class = st.selectbox("Cabin Class", options=cabin_class_options)
    submitted = st.form_submit_button("Show Forecast")

if submitted:
    condition = (
            (df['flightDate'].dt.date == flight_date) &
            (df['startingAirport'] == starting_airport) &
            (df['destinationAirport'] == destination_airport) &
            (df['isNonStop'] == is_nonstop) &
            (df['cabinClass'] == cabin_class)
    )
    filtered = df[condition].sort_values(by='searchDate')
    if filtered.empty:
        base_row = df.iloc[0]
    else:
        cheapest_row = filtered.loc[filtered['totalFare'].idxmin()]
        base_row = cheapest_row

    # For best offer, use the cleaned airline code to map to airline name and link.
    airline_code_clean = airline_code_dictionary[base_row['airlineCode']]
    mapping = {k.lower(): v for k, v in airline_mapping.items()}
    airline_info = mapping.get(airline_code_clean.lower(), (airline_code_clean, "Link not available"))
    cheapest_airline_name, booking_link = airline_info

    price_today = math.floor(base_row['totalFare'] / 10) * 10
    st.markdown("### Best Offer Today")
    st.write(f"**Cheapest Airline:** {cheapest_airline_name}")
    st.write(f"**Price Today:** ${price_today:.2f}")
    st.write(f"**Booking Link:** [Book Now]({booking_link})")



    historical, forecasted = autoregressive_forecast(base_row, model, model_features, n_forecast=7,
                                                     label_encoders=label_encoders)
    hist_days = list(range(-7, 0))
    forecast_days = list(range(1, 8))

    if forecasted[0] > price_today:
        diff = forecasted[0] - price_today
        st.markdown(
            f"**Alert:** The price is expected to rise by approximately ${diff:.2f} tomorrow. We recommend booking today!")
    elif forecasted[0] < price_today:
        diff = price_today - forecasted[0]
        st.markdown(
            f"**Good News:** The price is expected to drop by approximately ${diff:.2f} tomorrow. You might consider waiting for a better deal.")
    else:
        st.markdown("**Note:** The price is expected to stay the same.")

    try:
        plt.style.use('seaborn')
    except OSError:
        pass  # Fallback to default style if 'seaborn' is unavailable

    plt.figure(figsize=(12, 6))
    plt.plot(hist_days, historical, marker='o', markersize=8, linewidth=2, label='Historical Prices', color='blue')
    plt.plot(forecast_days, forecasted, marker='o', markersize=8, linewidth=2, linestyle='--', color='orange',
             label='Forecasted Prices')
    plt.fill_between(forecast_days, forecasted, alpha=0.3, color='orange')
    plt.xlabel("Day (relative)", fontsize=14)
    plt.ylabel("Total Fare", fontsize=14)
    plt.title("Historical Prices and 7-Day Forecast", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(plt)