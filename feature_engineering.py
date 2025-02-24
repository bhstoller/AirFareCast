import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from datetime import timedelta


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

    # Set Typing
    df["searchDate"] = pd.to_datetime(df["searchDate"])
    df["flightDate"] = pd.to_datetime(df["flightDate"])

    # Extract the travel duration in minutes
    df['travelDuration'] = extract_duration(df, 'travelDuration')
    
    # Impute missing travel distances
    df['travelDistance'] = impute_cols(SimpleImputer(strategy='mean'), 'totalTravelDistance', df)
    # imputer = SimpleImputer(strategy='mean')
    # df['totalTravelDistance'] = imputer.fit_transform(df[['totalTravelDistance']]).astype(int)

    # Extract departure times
    df = process_times(df, column_name='segmentsDepartureTimeRaw')
    
    # Extract departure hour and floating minutes
    df['departureTimeHour'] = df["segmentsDepartureTimeRaw"].dt.hour
    df['departureTimeFloat']= df["segmentsDepartureTimeRaw"].dt.hour + (df["segmentsDepartureTimeRaw"].dt.minute / 60)
    
    # Extract airline codes
    df = process_string_column(df, column_name='segmentsAirlineCode')

    # Extract and adjust cabin codes
    df = process_string_column(df, column_name='segmentsCabinCode')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('first', 'business')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('coach', 'economy')
    df["segmentsCabinCode"] = df["segmentsCabinCode"].str.replace('premium coach', 'premium economy')
    df.loc[df["isBasicEconomy"], 'segmentsCabinCode'] = 'basic economy'

    # Log transform skewed distributions (travelDuration and travelDistance)
    df['logTravelDuration'] = np.log(df["travelDuration"])
    df['logTravelDistance'] = np.log(df["totalTravelDistance"])

    # Bin seatsRemaining
    df["binnedSeatsRemaining"] = pd.cut(
        df["seatsRemaining"],
        bins=[-1, 2, 5, 9],
        labels=["Low", "Medium", "High"]
    )

    # Add days to departure feature
    df['daysToDeparture'] = (df["flightDate"] - df["searchDate"]).dt.days.astype(int)

    # Add day of the week features (departureDayOfWeek and isWeekend)
    df['departureDayOfWeek'] = df['flightDate'].dt.dayofweek
    df['isWeekend'] = df['departureDayOfWeek'].isin([5, 6])

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

    df.drop(columns=[
        'totalTravelDistance', 
        'isBasicEconomy', 
        'segmentsDepartureTimeRaw', 
        'segmentsArrivalTimeRaw'
    ], inplace= True)


    df = df.rename(columns= {
        "segmentsAirlineCode": "airlineCode", 
        "segmentsCabinCode": "cabinClass",
    }, inplace= True)

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
    df[column_name] = df[column_name].apply(
        lambda x: x.split('||')[0] if isinstance(x, str) else x
    )
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

    # Create the 'nearHoliday' column
    near_holiday = df['flightDate'].apply(lambda x: any((abs(x - holiday) <= timedelta(days=3)) for holiday in holiday_dates))

    return is_holiday, near_holiday