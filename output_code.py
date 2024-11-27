import holidays
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
import xgboost as xgb


def date_encoder(X, col="date"):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X["year"] = X[col].dt.year
    X["month"] = X[col].dt.month
    X["day"] = X[col].dt.day
    X["weekday"] = X[col].dt.weekday + 1
    X["hour"] = X[col].dt.hour

    X["is_weekend"] = np.where(
        X["weekday"] + 1 > 5, 1, 0
    )  # Binary variable indicating weekend or not (1=weekend, 0=weekday)

    fr_holidays = holidays.FR()  # Get list of FR holidays
    X["is_holiday"] = X[col].apply(
        lambda x: 1 if x in fr_holidays else 0
    )  # Binary variable indicating bank holiday or not (1 = holiday, 0 = not holiday)

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


# Import provided data
train = pd.read_parquet("/Users/pierrehaas/bike_counters/data/train.parquet")
test = pd.read_parquet("/Users/pierrehaas/bike_counters/data/final_test.parquet")

train = date_encoder(train, col="date")
test = date_encoder(test, col="date")

X_train = train.drop(columns=["bike_count", "log_bike_count"])
y_train = train["log_bike_count"]

X_test = test.copy()

pipe = Pipeline(
    steps=[
        ("preprocessor", TableVectorizer()),
        ("regressor", xgb.XGBRegressor()),
    ]
)

pipe.fit(X_train, y_train)

y_test = pipe.predict(X_test)

pd.DataFrame(y_test, columns=["log_bike_count"]).reset_index().rename(
    columns={"index": "Id"}
).to_csv("/Users/pierrehaas/bike_counters/predictions.csv", index=False)
