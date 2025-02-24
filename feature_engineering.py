import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from datetime import timedelta
from tqdm import tqdm
import time

# Enable tqdm for pandas operations
tqdm.pandas()

def apply_feature_engineering(df):
    """
    Apply feature engineering transformations to a DataFrame.
    :param df: pandas DataFrame
    :return: transformed DataFrame
    """
    print("Starting feature engineering...")
    start_time = time.time()

    print("Converting date columns...")
    df["searchDate"] = pd.to_datetime(df["searchDate"], errors="coerce")
    df["flightDate"] = pd.to_datetime(df["flightDate"], errors="coerce")

    print(f"Date conversion done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Extracting travel duration...")
    df["travelDuration"] = extract_duration(df, "travelDuration")
    print(f"Travel duration extraction done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Imputing missing travel distances...")
    df["travelDistance"] = impute_cols(SimpleImputer(strategy="mean"), "totalTravelDistance", df)
    print(f"Imputation done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Processing departure times...")
    df = process_times(df, column_name="segmentsDepartureTimeRaw")
    print(f"Departure time processing done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Extracting departure hour and float...")
    df["departureTimeHour"] = df["segmentsDepartureTimeRaw"].dt.hour
    df["departureTimeFloat"] = df["departureTimeHour"] + (df["segmentsDepartureTimeRaw"].dt.minute / 60)
    print(f"Departure time extraction done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Processing airline codes...")
    df = process_string_column(df, column_name="segmentsAirlineCode")
    print(f"Airline code processing done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Processing cabin codes...")
    df = process_string_column(df, column_name="segmentsCabinCode")
    df["segmentsCabinCode"] = df["segmentsCabinCode"].replace({
        "first": "business",
        "coach": "economy",
        "premium coach": "premium economy"
    })
    df.loc[df["isBasicEconomy"], "segmentsCabinCode"] = "basic economy"
    print(f"Cabin class processing done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Applying log transformations...")
    df["logTravelDuration"] = np.log(df["travelDuration"].replace(0, np.nan))  # Avoid log(0)
    df["logTravelDistance"] = np.log(df["totalTravelDistance"].replace(0, np.nan))  # Avoid log(0)
    print(f"Log transformations done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Binning seatsRemaining...")
    df["binnedSeatsRemaining"] = pd.cut(
        df["seatsRemaining"],
        bins=[-1, 2, 5, 9],
        labels=["Low", "Medium", "High"]
    )
    print(f"Seats binning done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Calculating days to departure...")
    df["daysToDeparture"] = (df["flightDate"] - df["searchDate"]).dt.days.astype(int)
    print(f"Days to departure calculation done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Extracting day of the week and weekend flags...")
    df["departureDayOfWeek"] = df["flightDate"].dt.dayofweek
    df["isWeekend"] = df["departureDayOfWeek"].isin([5, 6])
    print(f"Day of week processing done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Processing holiday features...")
    holiday_dates = pd.to_datetime([
        "2022-04-17", "2022-07-04", "2022-05-08",
        "2022-09-05", "2022-10-10", "2022-11-11"
    ])
    df["isHoliday"] = df["flightDate"].isin(holiday_dates)
    df["nearHoliday"] = df["flightDate"].apply(lambda x: any(abs(x - holiday) <= timedelta(days=3) for holiday in holiday_dates))
    print(f"Holiday processing done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Dropping unnecessary columns...")
    df.drop(columns=[
        "totalTravelDistance",
        "isBasicEconomy",
        "segmentsDepartureTimeRaw",
        "segmentsArrivalTimeRaw"
    ], inplace=True)
    print(f"Dropping columns done. Time elapsed: {time.time() - start_time:.2f}s")

    print("Renaming columns...")
    df.rename(columns={
        "segmentsAirlineCode": "airlineCode",
        "segmentsCabinCode": "cabinClass"
    }, inplace=True)
    print(f"Renaming done. Total time elapsed: {time.time() - start_time:.2f}s")

    print("Feature engineering complete!")
    return df


# Optimized helper functions:
def impute_cols(imputer, column, df):
    """
        Impute cols function.

        Parameters:
        - imputer: Description of the parameter.
        - column: Description of the parameter.
        - df: Description of the parameter.
    """
    df[column] = imputer.fit_transform(df[[column]])
    return df[column]

def extract_duration(df, column):
    """
        Extract duration function.

        Parameters:
        - df: Description of the parameter.
        - column: Description of the parameter.
    """
    extracted = df[column].str.extract(r'PT(?:(\d+)H)?(?:(\d+)M)?').fillna(0).astype(int)
    return extracted[0] * 60 + extracted[1]

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
    df[column_name] = pd.to_datetime(df[column_name].str.split('||').str[0], utc=True, errors="coerce")
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
    df[column_name] = df[column_name].str.split('||').str[0]
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