import pandas as pd

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