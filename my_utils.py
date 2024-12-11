# Standard Library Imports
import datetime
import io
import pickle
import re
import zipfile

# Third-Party Imports
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from skrub import DatetimeEncoder

# Define functions to process and merge the data

def date_encoder(X, col="date"):
    """
    Preprocesses a DataFrame by extracting and encoding various date-related features from a specified date column.
    
    Parameters:
    X (pd.DataFrame): The input DataFrame containing the date column.
    col (str): The name of the date column to be processed. Default is "date".
    
    Returns:
    pd.DataFrame: The updated DataFrame with new date-related features.
    """
    
    X = X.copy()  # modify a copy of X

    # Encode the date information from the DateOfDeparture columns
    X["year"] = X[col].dt.year
    X["quarter"] = X[col].dt.quarter
    X["month"] = X[col].dt.month
    X["month_scaled"] = np.cos(2 * np.pi * X["month"] / 12)
    X["weekofyear"] = X[col].dt.isocalendar().week
    X["day"] = X[col].dt.day
    X["weekday"] = X[col].dt.weekday + 1
    X["hour"] = X[col].dt.hour
    X["hour_scaled"] = np.cos(2 * np.pi * X["hour"] / 24)

    # Binary variable indicating weekend or not (1=weekend, 0=weekday)
    X["is_weekend"] = (X["weekday"] > 5).astype(int)

    # Binary variable indicating bank holiday or not (1=holiday, 0=not holiday)
    import holidays

    fr_bank_holidays = holidays.FR()  # Get list of FR holidays
    X["is_bank_holiday"] = X[col].apply(lambda x: 1 if x in fr_bank_holidays else 0)

    X = X.copy()  # modify a copy of X

    # Binary variable indicating school holiday or not (1=holiday, 0=not holiday)
    # https://www.data.gouv.fr/fr/datasets/vacances-scolaires-par-zones/
    fr_school_holidays = pd.read_csv(
        "/Users/pierrehaas/bike_counters/external_data/vacances_scolaires_france.csv"
    )[["date", "vacances_zone_c"]]

    # Ensure both DataFrames have a consistent datetime format
    X["date_normalized"] = pd.to_datetime(X[col]).dt.normalize()
    fr_school_holidays["date"] = pd.to_datetime(
        fr_school_holidays["date"]
    ).dt.normalize()

    # Create a dictionary from the holidays dataset for faster lookup
    holiday_mapping = dict(
        zip(fr_school_holidays["date"], fr_school_holidays["vacances_zone_c"])
    )

    # Map the normalized date to the holiday column
    X["is_school_holiday"] = (
        X["date_normalized"].map(holiday_mapping).fillna(0).astype(int)
    )

    # Drop the normalized date column if not needed
    X.drop(columns=["date_normalized"], inplace=True)

    # Finally, return the updated DataFrame
    return X

def train_test_processing(X_train, X_test):
    """
    Preprocesses the training and testing datasets by adding various features.

    This function includes two main preprocessing steps:
    1. Calculating the distance of each station from a reference station.
    2. Adding COVID-19 related features such as lockdown and curfew indicators.
    3. Clustering the stations based on the mean log_bike_count.

    Parameters:
    X_train (pd.DataFrame): The training dataset containing features including 'latitude', 'longitude', 'site_name', and 'date'.
    X_test (pd.DataFrame): The testing dataset containing features including 'latitude', 'longitude', 'site_name', and 'date'.

    Returns:
    tuple: A tuple containing the processed training and testing datasets (X_train, X_test) with additional features.
    """

    def dist_station(X):
        """
        Calculate the distance from a specific station to all other stations.

        This function computes the Euclidean distance between the coordinates of a specific station
        ("Totem 73 boulevard de Sébastopol") and all other stations in the dataset.

        Parameters:
        X (pd.DataFrame): A DataFrame containing the columns 'latitude', 'longitude', and 'site_name'.

        Returns:
        np.ndarray: An array of distances from the specified station to all other stations.
        """

        lat_long_station = (
            X[["latitude", "longitude"]]
            - mode(
                X[X["site_name"] == "Totem 73 boulevard de Sébastopol"][
                    ["latitude", "longitude"]
                ]
            )[0]
        )

        # We calculate the distance the previously identified site to all others
        # We call the result the distance to the center since the station is close to the center of Paris
        dist_station = np.linalg.norm(
            lat_long_station,
            axis=1,
        )

        return dist_station

    def covid_features(X):
        """
        Adds COVID-19 related features to the dataset.

        This includes a binary variable indicating the presence of a lockdown and curfew periods.
        Lockdown and curfew dates are based on historical data from France.

        Parameters:
        X (pd.DataFrame): The dataset containing a 'date' column.

        Returns:
        pd.DataFrame: The dataset with additional COVID-19 related features.
        """

        # Create a binary variable indicating the presence of a lockdown
        # https://fr.wikipedia.org/wiki/Chronologie_de_la_pand%C3%A9mie_de_Covid-19_en_France
        lockdown_dates = [
            ("2020-10-30", "2020-12-15"),
            ("2021-04-03", "2021-05-03"),
        ]

        X["covid_lockdown"] = 0

        for start, end in lockdown_dates:
            X.loc[(X["date"] >= start) & (X["date"] < end), "covid_lockdown"] = 1

        curfew_dates = [
            ("2020-10-17", "2020-10-30", 21, 6),  # 21h-6h
            ("2020-12-16", "2021-01-15", 20, 6),  # 20h-6h
            ("2021-01-15", "2021-03-19", 19, 6),  # 19h-6h
            ("2021-03-20", "2021-04-03", 18, 6),  # 18h-6h
            ("2021-05-03", "2021-06-09", 19, 6),  # 19h-6h
            ("2021-06-09", "2021-06-20", 23, 6),  # 23h-6h
        ]

        X["covid_curfew"] = 0

        for start_date, end_date, start_hour, end_hour in curfew_dates:
            X.loc[
                (X["date"] >= start_date)
                & (X["date"] < end_date)
                & (X["hour"] >= start_hour)
                & (X["hour"] <= end_hour),
                "covid_curfew",
            ] = 1

        return X

    X_train, X_test = covid_features(X_train), covid_features(X_test)

    X_train["dist_to_station"] = dist_station(X_train)
    X_test["dist_to_station"] = dist_station(X_test)

    # Group by counter_name and calculate the mean log_bike_count
    grouped_train = (
        X_train.groupby("counter_name", observed=True)["log_bike_count"]
        .mean()
        .reset_index()
    )

    # Reshape the data for clustering
    Y = grouped_train[["log_bike_count"]]

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=5)
    grouped_train["cluster"] = kmeans.fit_predict(Y)

    # Sort clusters by mean log_bike_count and reassign cluster labels
    sorted_clusters = (
        grouped_train.groupby("cluster")["log_bike_count"].mean().sort_values().index
    )
    cluster_mapping = {
        old_label: new_label for new_label, old_label in enumerate(sorted_clusters)
    }
    grouped_train["cluster"] = grouped_train["cluster"].map(cluster_mapping)

    # Merge the cluster labels back to the original DataFrame
    X_train = X_train.merge(
        grouped_train[["counter_name", "cluster"]], on="counter_name", how="left"
    )
    X_test = X_test.merge(
        grouped_train[["counter_name", "cluster"]], on="counter_name", how="left"
    )

    return X_train, X_test

def weather_processing(X, train_min, train_max, test_min, test_max):
    """
    Preprocesses weather data and applies PCA to extract principal components.

    This function performs the following steps:
    1. Drops unnecessary columns and groups the data by date.
    2. Calculates various weather-related features, including lagged and future values.
    3. Adds binary indicators for rain and maximum temperature for each day.
    4. Splits the data into training and testing sets based on provided date ranges.
    5. Applies PCA to reduce the dimensionality of the weather features.
    6. Merges the PCA components with additional weather-related features.

    Parameters:
    X (pd.DataFrame): The input weather dataset containing various weather-related columns.
    train_min (str): The start date for the training set (inclusive).
    train_max (str): The end date for the training set (inclusive).
    test_min (str): The start date for the testing set (inclusive).
    test_max (str): The end date for the testing set (inclusive).

    Returns:
    pd.DataFrame: A DataFrame containing the PCA components and additional weather-related features.
    """

    X_reduced = (
        X.drop(columns=["NUM_POSTE", "NOM_USUEL", "LAT", "LON", "QDXI3S"])
        .groupby("date")
        .mean()
        .dropna(axis=1, how="all")
        .interpolate(method="linear")
    )

    X_reduced["is_rain"] = (X_reduced["RR1"] > 0).astype(int)

    X_reduced["q_rain_lag_1h"] = X_reduced["RR1"].shift(1)
    X_reduced["t_rain_lag_1h"] = X_reduced["DRR1"].shift(1)
    X_reduced["is_rain_lag_1h"] = X_reduced["is_rain"].shift(1)

    X_reduced["q_rain_next_1h"] = X_reduced["RR1"].shift(-1)
    X_reduced["t_rain_next_1h"] = X_reduced["DRR1"].shift(-1)
    X_reduced["is_rain_next_1h"] = X_reduced["is_rain"].shift(-1)

    X_reduced["temp_lag_1h"] = X_reduced["T"].shift(1)
    X_reduced["temp_next_1h"] = X_reduced["T"].shift(-1)

    X_reduced["max_temp"] = X_reduced.groupby(X_reduced.index.date)["T"].transform(
        "max"
    )
    X_reduced["will_rain"] = (
        X_reduced.groupby(X_reduced.index.date)["RR1"]
        .transform(lambda x: (x > 0).any())
        .astype(int)
    )

    weather_features = [
        "RR1",
        "DRR1",
        "T",
        "TNSOL",
        "TCHAUSSEE",
        "U",
        "GLO",
        "q_rain_lag_1h",
        "t_rain_lag_1h",
        "q_rain_next_1h",
        "t_rain_next_1h",
        "temp_lag_1h",
        "temp_next_1h",
        "max_temp",
    ]

    X_reduced_train = X_reduced[
        (X_reduced.index >= train_min) & (X_reduced.index <= train_max)
    ]

    X_reduced_test = X_reduced[
        (X_reduced.index >= test_min) & (X_reduced.index <= test_max)
    ]

    n = 5
    pca = PCA(n_components=n)

    pca.fit(X_reduced_train[weather_features])

    X_pca_train = pca.transform(X_reduced_train[weather_features])
    X_pca_test = pca.transform(X_reduced_test[weather_features])

    X_pca_train = pd.DataFrame(
        X_pca_train,
        index=X_reduced_train[weather_features].index,
        columns=["weather_" + str(i) for i in range(1, n + 1)],
    ).reset_index()

    X_pca_test = pd.DataFrame(
        X_pca_test,
        index=X_reduced_test[weather_features].index,
        columns=["weather_" + str(i) for i in range(1, n + 1)],
    ).reset_index()

    X_pca = pd.concat([X_pca_train, X_pca_test], ignore_index=True)

    X_pca = X_pca.merge(
        X_reduced[(X_reduced.index >= train_min) & (X_reduced.index <= test_max)][
            ["is_rain", "is_rain_lag_1h", "is_rain_next_1h", "will_rain"]
        ],
        on="date",
    )

    return X_pca

