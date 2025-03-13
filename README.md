## Project Overview

This project focuses on analyzing airline fare pricing trends and predicting ticket prices using time-series forecasting and machine learning models. The goal is to provide insights that help travelers make cost-effective booking decisions by understanding fare patterns across different airlines, flight durations, cabin classes, and ticket availability.

### Demo
https://airfarecast-timeseriesfp.streamlit.app

### Key Objectives

•	Explore airline fare trends over time to identify pricing patterns.

•	Analyze key factors affecting airline pricing, such as seat availability, flight distance, and fare class.

•	Develop machine learning models to predict future ticket prices.

•	Provide data-driven recommendations for cost-effective flight bookings.

 Exploratory Data Analysis (EDA)

### Key Insights

1.	Fare Trends Over Time

•	Ticket prices fluctuate significantly, with a slight downward trend before departure.
•	Price spikes suggest high variability due to seasonal demand.
 
2.	Total Fare Distribution
 
•	Most fares are concentrated below $1,000, but outliers exceed $8,000, likely due to premium class tickets.
 
3.	Correlation Analysis
 
•	Base fare is the strongest predictor of total fare.
•	Seats remaining has little correlation with fare price, suggesting that availability alone does not drive ticket pricing.
 
4.	Pricing by Seat Availability
 
•	Prices do not significantly change based on the number of seats remaining.
•	Price variability exists across all seat availability categories.
 
5.	Non-Stop vs. Connecting Flights
 
•	Non-stop flights cost more on average, while connecting flights have greater variability in pricing.

6.	Basic Economy vs. Regular Economy

•	Basic Economy consistently has lower fares, while Regular Economy varies widely due to premium seating.
 
7.	Fare vs. Travel Distance
 
•	Longer flights generally cost more, but short-haul flights also have expensive fares, likely due to monopolized routes or last-minute bookings.

## Modeling Approach

### Models Tested

1.	Traditional Time Series Models
   
•	ARIMA & SARIMA: Failed to generalize well.

•	Prophet: Dropped due to poor performance.


2.	Machine Learning Models

•	Decision Tree
 
•	Random Forest (Best Model)

•	XGBoost

3.	Deep Learning Model

•	RNN-LSTM: Underperformed due to high variance in pricing data.
 
## Dataset

Dil Wong. (2022). Flight Prices [Data set]. Kaggle. https://www.kaggle.com/datasets/dilwong/flightprices
