# Standard Library Imports
import datetime
import io
import pickle
import re
import zipfile

# Third-Party Imports
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

# Import provided data
train = pd.read_parquet("/kaggle/input/train.parquet")
test = pd.read_parquet("/kaggle/input/final_test.parquet")

# Import additionally sourced data

# Import weather data
# https://meteo.data.gouv.fr/datasets/donnees-climatologiques-de-base-horaires/
weather = pd.read_csv(
    "/kaggle/input/weather-75000-2020-22.csv.gz",
    parse_dates=["AAAAMMJJHH"],
    date_format="%Y%m%d%H",
    compression="gzip",
    sep=";",
).rename(columns={"AAAAMMJJHH": "date"})


# Import public transport data
# https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/information/

# URLs of the zip files
urls = [
    "https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/files/e6bcf4c994951fc086e31db6819a3448/download/",
    "https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-ferre/files/e35b9ec0a183a8f2c7a8537dd43b124c/download/",
]

# Initialize an empty list to store DataFrames
dfs = []

# File matching pattern
pattern = r"data-rf-202\d/202\d_S\d+_NB_FER\.txt"

# Process each ZIP file
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get a list of all files in the archive and filter matching files
            matching_files = [f for f in z.namelist() if re.match(pattern, f)]

            # Read and concatenate the matching files
            for file in matching_files:
                with z.open(file) as f:
                    # Assuming the files are tab-separated and have a "JOUR" column
                    df = pd.read_csv(f, sep="\t", parse_dates=["JOUR"], dayfirst=True)
                    dfs.append(df)

# Combine all DataFrames
underground_transport = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-surface/information/

# URLs of the zip files
urls = [
    "https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-surface/files/41adcbd4216382c232ced4ccbf60187e/download/",
    "https://data.iledefrance-mobilites.fr/explore/dataset/histo-validations-reseau-surface/files/68cac32e8717f476905a60006a4dca26/download/",
]

# Initialize an empty list to store DataFrames
dfs = []

# File matching pattern
pattern = r"data-rs-202\d/202\d_T\d+_NB_SURFACE\.txt"

# Process each ZIP file
for url in urls:
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Get a list of all files in the archive and filter matching files
            matching_files = [f for f in z.namelist() if re.match(pattern, f)]

            # Read and concatenate the matching files
            for file in matching_files:
                with z.open(file) as f:
                    # Assuming the files are tab-separated and have a "JOUR" column
                    df = pd.read_csv(
                        f,
                        sep="\t",
                        parse_dates=["JOUR"],
                        dayfirst=True,
                        encoding="latin1",
                    )
                    dfs.append(df)

# Combine all DataFrames
overground_transport = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Import car traffic data
# https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/information/

# URLs of the zip files
urls = [
    "https://parisdata.opendatasoft.com/api/datasets/1.0/comptages-routiers-permanents-historique/attachments/opendata_txt_2020_zip/",
    # "https://parisdata.opendatasoft.com/api/datasets/1.0/comptages-routiers-permanents-historique/attachments/opendata_txt_2021_zip/", # This file's compression format is broken, thus I provide a download link below
    "https://www.dropbox.com/scl/fi/sfqzlzpyxcf4yied3yucc/comptage-routier-2021.zip?rlkey=6k6hr3kywl8tvm4ax1qv2nv88&st=ktehiium&dl=1",
]

# Initialize an empty list to store DataFrames
dfs = []

# Process each ZIP file
for i, url in enumerate(urls):
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Process every file in the archive
            for file in z.namelist():
                # Skip directories and __MACOSX files
                if file.endswith("/") or "__MACOSX" in file:
                    continue
                # For the second URL, ensure files are within "comptage-routier-2021" directory
                if i == 1 and not file.startswith("comptage-routier-2021/"):
                    continue
                with z.open(file) as f:
                    # Assuming the files are semicolon-separated and have a "t_1h" column
                    df = pd.read_csv(f, sep=";", parse_dates=["t_1h"])
                    dfs.append(df)

# Combine all DataFrames
cars_count = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# Import multi-modal transport data
# https://parisdata.opendatasoft.com/explore/dataset/comptage-multimodal-comptages/information/?disjunctive.label&disjunctive.mode&disjunctive.voie&disjunctive.sens&disjunctive.trajectoire&sort=-t&basemap=jawg.dark&location=13,48.87023,2.34614

# URL of the Parquet file
url = "https://parisdata.opendatasoft.com/api/explore/v2.1/catalog/datasets/comptage-multimodal-comptages/exports/parquet?lang=fr&timezone=Europe%2FParis"

# Send a GET request to download the file
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Load the content into a Pandas DataFrame
    parquet_file = io.BytesIO(response.content)
    multimodal_traffic = pd.read_parquet(parquet_file)


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


def transport_processing(X1, X2, train_min="2020-09-01", test_max="2021-10-18"):
    """
    Preprocesses transport data by aggregating daily validations and encoding date-related features.

    This function performs the following steps:
    1. Aggregates daily validation counts from two datasets.
    2. Combines the aggregated counts from both datasets.
    3. Encodes various date-related features using the `date_encoder` function.
    4. Filters the data based on the specified date range.

    Parameters:
    X1 (pd.DataFrame): The first dataset containing transport data with columns 'JOUR' (date) and 'NB_VALD' (number of validations).
    X2 (pd.DataFrame): The second dataset containing transport data with columns 'JOUR' (date) and 'NB_VALD' (number of validations).
    train_min (str): The start date for the training set (inclusive). Default is "2020-09-01".
    test_max (str): The end date for the testing set (inclusive). Default is "2021-10-18".

    Returns:
    pd.DataFrame: A DataFrame containing the aggregated and encoded transport data within the specified date range.
    """

    daily_X1 = X1.groupby("JOUR")["NB_VALD"].sum()
    daily_X2 = X2.groupby("JOUR")["NB_VALD"].sum()

    X = (daily_X1 + daily_X2).reset_index()

    X_reduced = date_encoder(X, col="JOUR")[
        (X["JOUR"] >= train_min) & (X["JOUR"] <= test_max)
    ]

    return X_reduced


def car_traffic_processing(X, train_min, test_max):
    """
    Preprocesses car traffic data by aggregating hourly traffic counts and calculating daily cumulative sums.

    This function performs the following steps:
    1. Aggregates hourly traffic counts.
    2. Calculates the cumulative sum of traffic counts for each day.
    3. Filters the data based on the specified date range.

    Parameters:
    X (pd.DataFrame): The input dataset containing car traffic data with columns 't_1h' (timestamp) and 'q' (traffic count).
    train_min (str): The start date for the training set (inclusive).
    test_max (str): The end date for the testing set (inclusive).

    Returns:
    pd.DataFrame: A DataFrame containing the aggregated and processed car traffic data within the specified date range.
    """
    X_hourly = X.groupby("t_1h")["q"].sum().reset_index()

    # Group by day and calculate the cumulative sum of 'q' for each day
    X_hourly["daily_cumsum"] = X_hourly.groupby(X_hourly["t_1h"].dt.to_period("d"))[
        "q"
    ].cumsum()

    return X_hourly[(X_hourly["t_1h"] >= train_min) & (X_hourly["t_1h"] <= test_max)]


def mm_traffic_processing(X, train_min, test_max):
    """
    Preprocesses multimodal traffic data by encoding dates, removing bike counts, and calculating cumulative sums.

    This function performs the following steps:
    1. Encodes dates appropriately by removing timezone information.
    2. Removes bike counts to avoid feeding the model with the target variable.
    3. Aggregates hourly traffic counts for each mode of transport.
    4. Calculates the cumulative sum of traffic counts for each vehicle type every day.
    5. Combines the original hourly counts with the cumulative sums.
    6. Filters the data based on the specified date range and fills missing values with zero.

    Parameters:
    X (pd.DataFrame): The input dataset containing traffic data with columns 't' (timestamp), 'mode' (mode of transport), and 'nb_usagers' (number of users).
    train_min (str): The start date for the training set (inclusive).
    test_max (str): The end date for the testing set (inclusive).

    Returns:
    pd.DataFrame: A DataFrame containing the processed traffic data within the specified date range, with additional features.
    """

    # Encode dates appropriately like train and test sets
    X["date"] = X["t"].dt.tz_localize(None)

    # Remove bike count to avoid feeding the model with the target variable
    mask = (X["mode"] == "Trottinettes + vélos") | (X["mode"] == "Vélos")
    hourly_vehicle_count = (
        X[~mask]
        .groupby(["date", "mode"])["nb_usagers"]
        .sum()
        .unstack()
        .drop(columns="van")
    )

    # Group by day and calculate the cumulative sum of 'nb_usagers' for each vehicle type every day
    daily_cumsum_vh_type = (
        hourly_vehicle_count.groupby(hourly_vehicle_count.index.date)
        .cumsum()
        .rename(columns=lambda x: x + "_cumsum")
    )

    mm_traffic_features = pd.concat(
        [
            hourly_vehicle_count,
            daily_cumsum_vh_type,
        ],
        axis=1,
    ).reset_index()

    return mm_traffic_features[
        (mm_traffic_features["date"] >= train_min)
        & (mm_traffic_features["date"] <= test_max)
    ].fillna(0)


def data_engineered(
    df_train,
    df_test,
    weather,
    underground_transport,
    overground_transport,
    cars_count,
    mm_traffic_count,
):
    """
    Preprocesses and merges various datasets to create engineered features for training and testing.

    This function performs the following steps:
    1. Encodes date-related features in the training and testing datasets.
    2. Merges weather data, public transport data, car traffic data, and multimodal traffic data with the training and testing datasets.

    Parameters:
    df_train (pd.DataFrame): The training dataset containing a 'date' column.
    df_test (pd.DataFrame): The testing dataset containing a 'date' column.
    weather (pd.DataFrame): The weather dataset containing weather-related features and a 'date' column.
    underground_transport (pd.DataFrame): The underground transport dataset containing transport data with columns 'year', 'month', 'day', and 'NB_VALD'.
    overground_transport (pd.DataFrame): The overground transport dataset containing transport data with columns 'year', 'month', 'day', and 'NB_VALD'.
    cars_count (pd.DataFrame): The car traffic dataset containing traffic data with columns 't_1h' (timestamp) and 'q' (traffic count).
    mm_traffic_count (pd.DataFrame): The multimodal traffic dataset containing traffic data with columns 'date', 'mode', and 'nb_usagers'.

    Returns:
    tuple: A tuple containing the processed training and testing datasets (df_train, df_test) with merged features.
    """

    def merging_data(data, weather, public_transport, car_traffic, mm_traffic):
        """
        Merges weather, public transport, car traffic, and multimodal traffic data with the main dataset.

        Parameters:
        data (pd.DataFrame): The main dataset to merge with other datasets.
        weather (pd.DataFrame): The weather dataset containing weather-related features and a 'date' column.
        public_transport (pd.DataFrame): The public transport dataset containing transport data with columns 'year', 'month', 'day', and 'NB_VALD'.
        car_traffic (pd.DataFrame): The car traffic dataset containing traffic data with columns 't_1h' (timestamp) and 'q' (traffic count).
        mm_traffic (pd.DataFrame): The multimodal traffic dataset containing traffic data with columns 'date', 'mode', and 'nb_usagers'.

        Returns:
        pd.DataFrame: The merged dataset with additional features.
        """

        # Merge weather data
        data = data.merge(weather, on="date", how="left")

        # Merge public transport data
        data = data.merge(
            public_transport[["year", "month", "day", "NB_VALD"]],
            on=["year", "month", "day"],
            how="left",
        )

        # Merge car traffic data
        data = (
            data.merge(car_traffic, left_on="date", right_on="t_1h", how="left")
            .drop(columns=["t_1h"])
            .dropna()
            .reset_index(drop=True)
        )

        data = data.merge(mm_traffic, on="date", how="left")

        return data

    train_min, train_max = df_train["date"].min(), df_train["date"].max()
    test_min, test_max = df_test["date"].min(), df_test["date"].max()

    # Encoding the date
    df_train, df_test = date_encoder(df_train), date_encoder(df_test)

    # Processing the data
    df_train, df_test = train_test_processing(df_train, df_test)

    # Processing weather data
    weather_processed = weather_processing(
        weather, train_min, train_max, test_min, test_max
    )

    # Processing transport data
    transport_processed = transport_processing(
        underground_transport, overground_transport
    )

    # Processing car traffic data
    new_index = df_train["date"].unique().tolist() + df_test["date"].unique().tolist()
    car_traffic_processed = (
        car_traffic_processing(cars_count, train_min, test_max)
        .set_index("t_1h")
        .reindex(new_index)
        .interpolate(method="linear")
        .reset_index()
    )

    # Processing multimodal traffic data
    mm_traffic_processed = mm_traffic_processing(mm_traffic_count, train_min, test_max)

    # Merging the data
    df_train = merging_data(
        df_train,
        weather_processed,
        transport_processed,
        car_traffic_processed,
        mm_traffic_processed,
    )

    df_test = merging_data(
        df_test,
        weather_processed,
        transport_processed,
        car_traffic_processed,
        mm_traffic_processed,
    )

    return df_train, df_test


df_train, X_test = data_engineered(
    train,
    test,
    weather,
    underground_transport,
    overground_transport,
    cars_count,
    multimodal_traffic,
)

df_train["total_seconds"] = DatetimeEncoder().fit_transform(df_train["date"])[
    "date_total_seconds"
]
X_test["total_seconds"] = DatetimeEncoder().fit_transform(X_test["date"])[
    "date_total_seconds"
]

df_train.sort_values(["date", "counter_id"], inplace=True)

y_train = df_train["log_bike_count"]

X_train = df_train.drop(
    columns=[
        "date",
        "counter_name",
        "site_name",
        "counter_installation_date",
        "bike_count",
        "log_bike_count",
    ],
)

X_test = X_test.drop(
    columns=[
        "date",
        "counter_name",
        "site_name",
        "counter_installation_date",
    ],
)

cols = [
    "2 roues motorisées",
    "Autobus et autocars",
    "Trottinettes",
    "Véhicules lourds > 3,5t",
    "Véhicules légers < 3,5t",
    "2 roues motorisées_cumsum",
    "Autobus et autocars_cumsum",
    "Trottinettes_cumsum",
    "Véhicules lourds > 3,5t_cumsum",
    "Véhicules légers < 3,5t_cumsum",
]

X_train[cols] = X_train[cols].fillna(0)

cols_to_label_encode = [
    "counter_id",
    "site_id",
    "coordinates",
    "counter_technical_id",
]

# Define the preprocessor
preprocessor = ColumnTransformer(
    [
        ("label_encoder", OrdinalEncoder(), cols_to_label_encode),
    ],
    remainder="passthrough",
)

# Fit and transform the data
preprocessor.fit(X_train)
X_train_enc = preprocessor.transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# Get feature names from the transformers
label_encoder_feature_names = [f"{col}_encoded" for col in cols_to_label_encode]
passthrough_feature_names = [
    col for col in X_train.columns if col not in cols_to_label_encode
]

# Combine all feature names
all_feature_names = label_encoder_feature_names + passthrough_feature_names

# Convert the numpy ndarray to a pandas DataFrame
X_train_enc_df = pd.DataFrame(X_train_enc, columns=all_feature_names)
X_test_enc_df = pd.DataFrame(X_test_enc, columns=all_feature_names)

# Convert data types explicitly
X_train_enc_df = X_train_enc_df.apply(pd.to_numeric, errors="coerce")
X_test_enc_df = X_test_enc_df.apply(pd.to_numeric, errors="coerce")

model = ExtraTreesRegressor()

model.fit(
    X_train_enc_df,
    y_train,
)

y_pred = model.predict(X_test_enc_df)

pd.DataFrame(y_pred, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("submission.csv", index=False)
