import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from datetime import timedelta
from tqdm import tqdm
import time
import warnings
warnings.simplefilter(action='ignore', category=Warning)

def apply_feature_engineering(df):
    """
        Apply feature engineering function.
        
        Parameters:
        - df: Description of the parameter.
    """
    """
    Apply feature engineering transformations to a DataFrame.
    :param df: pandas DataFrame
    :return: transformed DataFrame
    """

    print("Starting feature engineering...")
    start_time = time.time()

    print("Converting date columns...")

    # Set Typing
    df["searchDate"] = pd.to_datetime(df["searchDate"])
    df["flightDate"] = pd.to_datetime(df["flightDate"])

    print(f"Date conversion done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Extracting travel duration...")

    # Extract the travel duration in minutes
    df['travelDuration'] = extract_duration(df, 'travelDuration')

    print(f"Travel duration extraction done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Imputing missing travel distances...")
    
    # Impute missing travel distances
    df['travelDistance'] = impute_cols(SimpleImputer(strategy='mean'), 'totalTravelDistance', df)

    print(f"Imputation done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Processing departure times...")

    # Extract departure times
    df = process_times(df, column_name='segmentsDepartureTimeRaw')

    print(f"Departure time processing done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Extracting departure hour and float...")
    
    # Extract departure hour and floating minutes
    df['departureTimeHour'] = df["segmentsDepartureTimeRaw"].dt.hour
    df['departureTimeFloat']= df["segmentsDepartureTimeRaw"].dt.hour + (df["segmentsDepartureTimeRaw"].dt.minute / 60)
    
    print(f"Departure time extraction done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Processing airline codes...")

    # Extract airline codes
    df = process_string_column(df, column_name='segmentsAirlineCode')

    print(f"Airline code processing done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Processing cabin codes...")

    # Extract and adjust cabin codes
    df = process_string_column(df, column_name='segmentsCabinCode')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('first', 'business')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('coach', 'economy')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('premium coach', 'premium economy')
    df.loc[df["isBasicEconomy"], 'segmentsCabinCode'] = 'basic economy'

    print(f"Cabin class processing done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Binning seatsRemaining...")

    # Bin seatsRemaining
    df["binnedSeatsRemaining"] = pd.cut(
        df["seatsRemaining"],
        bins=[-1, 2, 5, 9],
        labels=[0, 1, 2]
    )

    print(f"Seats binning done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Calculating days to departure...")

    # Add days to departure feature
    df['daysToDeparture'] = (df["flightDate"] - df["searchDate"]).dt.days.astype(int)

    # Add day of the week features (departureDayOfWeek and isWeekend)
    df['departureDayOfWeek'] = df['flightDate'].dt.dayofweek
    df['isWeekend'] = df['departureDayOfWeek'].isin([5, 6])

    print(f"Day of week processing done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Processing holiday features...")

    # Add holiday features (isHoliday and nearHoliday)
    holiday_dates = {
        "Easter Sunday": "2022-04-17",
        "Independence Day": "2022-07-04",
        "Mother's Day": "2022-05-08",
        "Labor Day": "2022-09-05",
        "Columbus Day": "2022-10-10",
        "Veterans Day": "2022-11-11"
    }

    df['isHoliday'], df['nearHoliday'] = add_holidays(holiday_dates, df)

    print(f"Holiday features processing done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Dropping columns...")

    df.drop(columns=[ 
        'isBasicEconomy', 
        'segmentsDepartureTimeRaw',
        'totalTravelDistance',
        'searchDate',
        'flightDate' 
    ], inplace= True)

    print(f"Dropping columns done. Time elapsed: {time.time() - start_time:.2f}s")
    print("Renaming columns...")

    df.rename(columns= {
        "segmentsAirlineCode": "airlineCode", 
        "segmentsCabinCode": "cabinClass",
    }, inplace= True)

    print(f"Renaming done. Total time elapsed: {time.time() - start_time:.2f}s")
    print("Adding dummies...")

    dummy_cols = [
        'startingAirport',
        'destinationAirport',
        'airlineCode',
        'cabinClass',
        'binnedSeatsRemaining'
    ]
    df = add_dummies(df, cols=dummy_cols)

    print(f"Dummies added. Total time elapsed: {time.time() - start_time:.2f}s")
    print("Feature engineering complete!")
    return df

def impute_cols(imputer, column, df):
    """
        Impute cols function.
        
        Parameters:
        - imputer: Description of the parameter.
        - column: Description of the parameter.
        - df: Description of the parameter.
    """
    return imputer.fit_transform(df[[column]]).astype(int)

def extract_duration(df, column):
    """
        Extract duration function.
        
        Parameters:
        - df: Description of the parameter.
        - column: Description of the parameter.
    """
    extracted = df[column].str.extract(r'PT(?:(\d+)H)?(?:(\d+)M)?').fillna(0).astype(int)
    return extracted[0].to_numpy() * 60 + extracted[1].to_numpy()

def process_times(df, column_name):
    """
    Converts a flight time column to datetime.
    - Extracts only the first timestamp if multiple exist (separated by "||").
    - Converts to `datetime64[ns]`, ensuring proper timezone handling.
    - If timezone-aware, converts to UTC and removes the timezone.

    Parameters:
    df (pd.DataFrame): The dataframe.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: The updated dataframe with a converted datetime column.
    """
    df[column_name] = df[column_name].apply(
        lambda x: pd.to_datetime(x.split('||')[0], utc=True) if isinstance(x, str) else pd.NaT
    )

    # Convert timezone-aware timestamps to naive datetime (remove timezone)
    df[column_name] = df[column_name].dt.tz_localize(None)

    return df

def process_string_column(df, column_name):
    """
    Processes a column containing strings with "||".

    - If there is no "||", keeps the original string.
    - If there are multiple values separated by "||", extracts only the first value.

    Parameters:
    df (pd.DataFrame): The dataframe.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: The updated dataframe with processed strings.
    """
    df[column_name] = df[column_name].astype(str).str.split('||').str[0]
    return df

def add_holidays(holiday_dates, df):
    """
    Add holidays function.
    
    Parameters:
    - holiday_dates: Description of the parameter.
    - df: Description of the parameter.
    """
    # Convert to datetime
    holiday_dates = pd.to_datetime(list(holiday_dates.values()))

    # Create the 'isHoliday' column
    is_holiday = df['flightDate'].isin(holiday_dates)

    # Convert holiday_dates to a NumPy array for efficient broadcasting
    holiday_array = np.array(holiday_dates)

    # Compute absolute differences in a vectorized way and check the condition
    near_holiday = (np.abs(df["flightDate"].values[:, None] - holiday_array) <= np.timedelta64(3, 'D')).any(axis=1)
    
    return is_holiday, near_holiday

def add_dummies(df, cols):
    df = pd.get_dummies(df, columns= cols, drop_first=True)
    return df