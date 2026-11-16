
import pandas as pd


DATA_PATH = "data/raw/household_power_consumption.txt"


def load_and_preprocess():

    # 1. Load the raw dataset
    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values="?",
        low_memory=False
    )

    print("Original dataset shape:", df.shape)

    # 2. Combine Date and Time into one datetime column
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S"
    )

    # 3. Set datetime as the index
    df.set_index("datetime", inplace=True)

    # 4. Remove the original Date and Time columns
    df.drop(columns=["Date", "Time"], inplace=True)

    # 5. Convert all columns to numeric values
    df = df.apply(pd.to_numeric, errors="coerce")

    # 6. Handle missing values
    df = df.interpolate(method="time")

    # 7. Convert minute-level data into hourly averages
    hourly_data = df.resample("h").mean()

    print("Hourly dataset shape:", hourly_data.shape)

    # 8. Create the target: next hour's electricity consumption
    hourly_data["target"] = (
        hourly_data["Global_active_power"].shift(-1)
    )

    # 9. Create lag features
    power = hourly_data["Global_active_power"]

    hourly_data["previous_hour"] = power.shift(1)
    hourly_data["previous_2_hours"] = power.shift(2)
    hourly_data["previous_3_hours"] = power.shift(3)
    hourly_data["previous_6_hours"] = power.shift(6)
    hourly_data["previous_day"] = power.shift(24)
    hourly_data["previous_week"] = power.shift(168)

    # 10. Create rolling features
    hourly_data["rolling_mean_3"] = power.rolling(3).mean()
    hourly_data["rolling_mean_6"] = power.rolling(6).mean()
    hourly_data["rolling_mean_24"] = power.rolling(24).mean()
    hourly_data["rolling_std_24"] = power.rolling(24).std()

    # 11. Create calendar features
    hourly_data["hour"] = hourly_data.index.hour
    hourly_data["day_of_week"] = hourly_data.index.dayofweek
    hourly_data["month"] = hourly_data.index.month
    hourly_data["is_weekend"] = (
        hourly_data["day_of_week"] >= 5
    ).astype(int)

    # 12. Remove rows with missing feature or target values
    hourly_data.dropna(inplace=True)

    # 13. Save processed dataset
    hourly_data.to_csv("data/processed/hourly_features.csv")

    print("Final processed dataset shape:", hourly_data.shape)
    print("\nFirst five rows:")
    print(hourly_data.head())

    return hourly_data


if __name__ == "__main__":
    load_and_preprocess()
