import numpy as np
import pandas as pd
import re
import streamlit as st
from datetime import date
from main import get_travel_info
from main import TRAVEL_DATA  # Importing travel data dictionary

# List of unique airport codes
AIRPORTS = sorted(set([airport for route in TRAVEL_DATA.keys() for airport in route]))

# Streamlit UI
st.title("Flight Information Input")

# Dropdowns for selecting departure and arrival locations
departure = st.selectbox("Departure Location", AIRPORTS)
arrival = st.selectbox("Arrival Location", [airport for airport in AIRPORTS if airport != departure])

# Checkboxes for refundable and nonstop options
is_refundable = st.checkbox("Refundable Ticket")
is_nonstop = st.checkbox("Nonstop Flight")

# Date and time input for departure
departure_date = st.date_input("Departure Date", min_value=date.today())

# User enters time
departure_time = st.text_input("Enter Departure Time (HH:MM)", "12:00", key="departure_time_input")

# Validate time format
time_pattern = r"^(0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$"
if departure_time and not re.match(time_pattern, departure_time):
    st.error("Invalid time format. Use HH:MM (24-hour format).")

# Numeric input for remaining seats
seats_remaining = st.number_input("Seats Remaining", min_value=0, step=1)

# Dropdown for cabin class
cabin_class = st.selectbox("Cabin Class", ["Economy", "Basic Economy", "Premium Economy", "Business"])

# Button to submit form
if st.button("Submit"):
    st.success("Flight information submitted successfully!")

