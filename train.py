
import pandas as pd
import numpy as np
import mlflow
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load processed dataset
DATA_PATH = "data/processed/hourly_features.csv"

df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)

# 2. Select features used by the model
FEATURES = [
    "previous_hour",
    "previous_2_hours",
    "previous_3_hours",
    "previous_6_hours",
    "previous_day",
    "previous_week",
    "rolling_mean_3",
    "rolling_mean_6",
    "rolling_mean_24",
    "rolling_std_24",
    "hour",
    "day_of_week",
    "month",
    "is_weekend"
]

TARGET = "target"

X = df[FEATURES]
y = df[TARGET]

# 3. Chronological 80:20 train-test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 4. Configure MLflow
mlflow.set_experiment("Electricity-Consumption-Prediction")

results = []
best_mae = float("inf")
best_model = None
best_model_name = None


# 5. Function to train and evaluate a model
def train_and_evaluate(model, model_name, parameters):

    global best_mae, best_model, best_model_name

    with mlflow.start_run(run_name=model_name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Record experiment details in MLflow
        mlflow.log_param("algorithm", model_name)

        for key, value in parameters.items():
            mlflow.log_param(key, value)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        print("\nModel:", model_name)
        print("MAE:", mae)
        print("RMSE:", rmse)
        print("R2:", r2)

        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        # Select the model with the lowest MAE
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_model_name = model_name


# 6. Experiment 1: Linear Regression
train_and_evaluate(
    LinearRegression(),
    "Linear-Regression",
    {}
)

# 7. Experiment 2: Random Forest
train_and_evaluate(
    RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),
    "Random-Forest-100",
    {
        "n_estimators": 100,
        "max_depth": 15
    }
)

# 8. Experiment 3: Random Forest with different settings
train_and_evaluate(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    ),
    "Random-Forest-200",
    {
        "n_estimators": 200,
        "max_depth": 20
    }
)


# 9. Display model comparison
results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# 10. Save the best model
joblib.dump(
    {
        "model": best_model,
        "features": FEATURES,
        "model_name": best_model_name
    },
    "model/model.joblib"
)

print("\nBest model:", best_model_name)
print("Best MAE:", best_mae)
print("Model saved successfully!")
