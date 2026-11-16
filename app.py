
import logging
import math
import statistics
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------
# 1. Load the saved ML model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

artifact = joblib.load(BASE_DIR / "model" / "model.joblib")

model = artifact["model"]
FEATURES = artifact["features"]
MODEL_NAME = artifact["model_name"]


# --------------------------------------------------
# 2. Configure prediction logging
# --------------------------------------------------

(BASE_DIR / "logs").mkdir(exist_ok=True)

logging.basicConfig(
    filename=BASE_DIR / "logs" / "predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 3. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Electricity Consumption Prediction API",
    description="Predicts next-hour electricity consumption.",
    version="1.0"
)


# --------------------------------------------------
# 4. Define and validate user input
# --------------------------------------------------

class PredictionInput(BaseModel):

    history: list[float] = Field(
        ...,
        min_length=168,
        description="At least 168 hourly electricity readings in kW, oldest first."
    )

    last_timestamp: datetime = Field(
        ...,
        description="Timestamp of the most recent hourly reading."
    )

    @field_validator("history")
    @classmethod
    def validate_history(cls, values):

        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(
                "Readings must be finite, non-negative numbers."
            )

        return values


# --------------------------------------------------
# 5. Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": MODEL_NAME
    }


# --------------------------------------------------
# 6. Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return FileResponse(BASE_DIR / "static" / "index.html")


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(input_data: PredictionInput):

    # Use the latest 168 hourly readings
    history = input_data.history[-168:]

    # Timestamp of the hour we want to predict
    next_timestamp = (
        input_data.last_timestamp + timedelta(hours=1)
    )

    # Calculate the 14 features used during training
    features = {

        "previous_hour": history[-1],
        "previous_2_hours": history[-2],
        "previous_3_hours": history[-3],
        "previous_6_hours": history[-6],
        "previous_day": history[-24],
        "previous_week": history[-168],

        "rolling_mean_3": statistics.mean(history[-3:]),
        "rolling_mean_6": statistics.mean(history[-6:]),
        "rolling_mean_24": statistics.mean(history[-24:]),

        "rolling_std_24": statistics.stdev(history[-24:]),

        "hour": input_data.last_timestamp.hour,
        "day_of_week": input_data.last_timestamp.weekday(),
        "month": input_data.last_timestamp.month,
        "is_weekend": int(input_data.last_timestamp.weekday() >= 5)
    }

    # Keep feature order identical to model training
    input_df = pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES
    )

    # Generate prediction
    prediction = float(model.predict(input_df)[0])

    # Calculate recent average for comparison
    recent_average = statistics.mean(history[-24:])

    # Rule-based suggestions
    suggestions = []

    if prediction > recent_average * 1.2:

        suggestions.append(
            "Predicted consumption is more than 20% above "
            "the recent 24-hour average. Check high-power appliances."
        )

    elif prediction > recent_average:

        suggestions.append(
            "Predicted consumption is above the recent average. "
            "Consider shifting flexible appliance usage."
        )

    else:

        suggestions.append(
            "Predicted consumption is within or below "
            "the recent 24-hour average."
        )

    suggestions.append(
        "Monitor hourly consumption to identify unusual usage patterns."
    )

    # Log prediction input and output
    logger.info(
        "Input: %s | Prediction: %.4f kW",
        input_data.model_dump_json(),
        prediction
    )

    return {
        "predicted_consumption_kw": round(prediction, 4),
        "prediction_for": next_timestamp.isoformat(),
        "model_used": MODEL_NAME,
        "recent_24_hour_average_kw": round(recent_average, 4),
        "suggestions": suggestions
    }
