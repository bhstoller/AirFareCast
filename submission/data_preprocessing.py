import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
from tqdm import tqdm
import time
import warnings
import pickle
from tqdm.notebook import tqdm
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

warnings.simplefilter(action='ignore', category=Warning)

def process_data():
    """
    Get the loaded, feature engineered, and wrangled dataset.
    """
    df = load_data()
    df = apply_feature_engineering(df)
    df = add_historical_price_features(df)

    return df

# Read the data using parquet
def load_data():
    """
    Load the dataset using parquet and pyarrow
    """
    df = pd.read_parquet(
        "data/itineraries_snappy.parquet", 
        engine= "pyarrow", 
        columns= [
            "searchDate", 
            "flightDate", 
            "startingAirport", 
            "destinationAirport",
            "travelDuration", 
            "isBasicEconomy", 
            "isRefundable", 
            "isNonStop", 
            "totalFare", 
            "seatsRemaining", 
            "totalTravelDistance",
            "segmentsDepartureTimeRaw", 
            "segmentsAirlineCode", 
            "segmentsCabinCode"
        ]
    )
    return df

def apply_feature_engineering(df, dnn= False, drop_search_date= True):
    print("Starting feature engineering...")

    df = df.copy()

    print("Converting date columns...")
    df["searchDate"] = pd.to_datetime(df["searchDate"], errors='coerce')
    df["flightDate"] = pd.to_datetime(df["flightDate"], errors='coerce')

    print("Extracting travel duration...")
    df['travelDuration'] = extract_duration(df, 'travelDuration')

    print("Imputing missing travel distances...")
    imputer = SimpleImputer(strategy='mean')
    df['travelDistance'] = imputer.fit_transform(df[['totalTravelDistance']])

    print("Processing departure times...")
    df['departureTimeHour'] = df["segmentsDepartureTimeRaw"].str[11:13]

    print("Processing airline and cabin class codes...")
    df["segmentsCabinCode"] = df["segmentsCabinCode"].astype(str).copy()
    df["segmentsCabinCode"] = df["segmentsCabinCode"].replace({
        'first': 'business',
        'coach': 'economy',
        'premium coach': 'premium economy'
    })
    df.loc[df["isBasicEconomy"].astype(bool), 'segmentsCabinCode'] = 'basic economy'
    df.rename(columns={"segmentsAirlineCode": "airlineCode", "segmentsCabinCode": "cabinClass"}, inplace=True)

    print("Applying Label Encoding...")
    label_encoders = {}
    categorical_cols = ['airlineCode', 'cabinClass', 'startingAirport', 'destinationAirport']

    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Save label encoders to file for future use
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    print("Label Encoding complete!")

    print("Calculating days to departure...")
    df['daysToDeparture'] = (df["flightDate"] - df["searchDate"]).dt.days.astype('Int64')
    df['departureDayOfWeek'] = df['flightDate'].dt.dayofweek
    df['isWeekend'] = df['departureDayOfWeek'].isin([5, 6])

    print("Processing holiday features...")
    holiday_dates = {
        "Easter Sunday": "2022-04-17",
        "Independence Day": "2022-07-04",
        "Mother's Day": "2022-05-08",
        "Labor Day": "2022-09-05",
        "Columbus Day": "2022-10-10",
        "Veterans Day": "2022-11-11"
    }
    df['isHoliday'], df['nearHoliday'] = add_holidays(holiday_dates, df)

    print("Dropping unnecessary columns...")
    drop_columns = [
        'isBasicEconomy', 
        'segmentsDepartureTimeRaw', 
        'totalTravelDistance',
    ]

    if drop_search_date:
        drop_columns.append('searchDate')

    if not dnn:
        drop_columns += [
            'flightDate', 
            'airlineCode',
            'cabinClass'
        ]

    df.drop(columns=drop_columns, inplace=True, errors='ignore')

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

# def process_string_column(df, column_name):
#     """
#     Processes a column containing strings with "||".

#     - If there is no "||", keeps the original string.
#     - If there are multiple values separated by "||", extracts only the first value.

#     Parameters:
#     df (pd.DataFrame): The dataframe.
#     column_name (str): The name of the column to process.

#     Returns:
#     pd.DataFrame: The updated dataframe with processed strings.
#     """
#     df[column_name] = df[column_name].astype(str).str.split('||').str[0]
#     return df

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

def add_historical_price_features(df, num_lags=7):
    """
    Adds historical price features (t-1, t-2, ..., t-7) with visual feedback on slow operations.
    Uses Dask to sort the DataFrame with progress feedback and pandas groupby+shift for lag features.
    """
    df = df.copy()

    # Convert date columns
    df['flightDate'] = pd.to_datetime(df['flightDate'])
    df['searchDate'] = pd.to_datetime(df['searchDate'])

    # Use Dask for faster sorting on large DataFrames.
    ddf = dd.from_pandas(df, npartitions=8)
    sort_columns = ['flightDate', 'departureTimeHour', 'startingAirport', 'destinationAirport', 'searchDate']

    print("Starting sort operation...")
    with ProgressBar():
        ddf = ddf.sort_values(by=sort_columns)
        ddf = ddf.compute()
    df = ddf
    print("Sort completed.")

    group_cols = ['flightDate', 'departureTimeHour', 'startingAirport', 'destinationAirport']

    for lag in tqdm(range(1, num_lags + 1), desc="Creating lag features"):
        df[f'price_t_minus_{lag}'] = df.groupby(group_cols)['totalFare'].shift(lag)

    print("Historical price features added successfully!")
    return df