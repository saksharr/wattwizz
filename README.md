# ⚡ WattWise – Electricity Consumption Prediction

An AI/ML-powered web application that predicts future household electricity consumption using historical electricity usage data. WattWise uses machine learning to forecast the next hour's consumption and provides insights to help users understand their energy usage.

## 🌐 Project Links

- **Live Application:** https://wattwizz-p2y7.onrender.com
- **GitHub Repository:** https://github.com/saksharr/wattwizz
- **API Documentation:** https://wattwizz-p2y7.onrender.com/docs

---

## 1. Problem Statement

Electricity consumption varies depending on household usage patterns, time of day and other factors. Understanding these patterns can help households monitor and manage their electricity usage.

WattWizz addresses this problem by using historical electricity consumption data to predict future consumption. It provides an interactive dashboard where users can submit historical readings and receive predictions along with energy-saving suggestions.

## 2. Dataset

**Dataset:** Individual Household Electric Power Consumption

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains household electricity measurements collected over time.

Important features include:

- Global_active_power
- Global_reactive_power
- Voltage
- Global_intensity
- Sub_metering_1
- Sub_metering_2
- Sub_metering_3

The data is cleaned, converted into a time-series format and aggregated into hourly readings.

## 3. Data Preprocessing and Feature Engineering

The following preprocessing steps are performed:

1. Handling missing and invalid values.
2. Converting date and time information into datetime format.
3. Converting relevant columns into numeric values.
4. Sorting records chronologically.
5. Aggregating electricity consumption into hourly readings.
6. Creating time-series features for model training.

### Feature Engineering

The model uses the following features:

- **Lag features:** Previous hour, previous 2 hours, previous 3 hours, previous 6 hours, previous day and previous week.
- **Rolling features:** Rolling mean for 3, 6 and 24 hours, and rolling standard deviation for 24 hours.
- **Time features:** Hour, day of week, month and weekend indicator.

These features help the model identify historical consumption patterns and temporal trends.

## 4. Machine Learning Models

This project uses regression algorithms because electricity consumption is a continuous numerical value.

The following models and configurations are compared:

| Experiment | Algorithm | Configuration |
|---|---|---|
| 1 | Linear Regression | Default parameters |
| 2 | Random Forest Regressor | 100 estimators, max depth 15 |
| 3 | Random Forest Regressor | 200 estimators, max depth 20 |

### Evaluation Metrics

The models are evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error):** Measures prediction error while penalizing larger errors.
- **R² Score:** Measures how much variation in the target variable is explained by the model.

### Experiment Tracking with MLflow

MLflow is used to track model experiments, parameters and evaluation metrics.

- Experiment name: `Electricity-Consumption-Prediction`
- Total tracked runs: 3
- Algorithms evaluated: 2
- Logged metrics: MAE, RMSE and R²
- Logged parameters: Algorithm and model hyperparameters.

The model with the lowest MAE is selected as the best-performing candidate.

## 5. System Architecture

The application follows this workflow:

```text
Electricity Consumption Dataset
             |
             v
     Data Preprocessing
             |
             v
    Feature Engineering
             |
             v
      Model Training
             |
             v
     MLflow Experiment Tracking
             |
             v
       Model Selection
             |
             v
     Saved Model (joblib)
             |
             v
       FastAPI Backend
             |
             v
       HTML/CSS/JS Frontend
             |
             v
       Docker Container
             |
             v
       Render Deployment
```

## 6. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| Model Serialization | Joblib |
| Backend | FastAPI |
| Data Validation | Pydantic |
| Frontend | HTML, CSS, JavaScript |
| Visualization | JavaScript Canvas |
| Testing | Pytest |
| Containerization | Docker |
| Deployment | Render |
| Version Control | Git and GitHub |

## 7. Application Features

- Interactive electricity consumption dashboard.
- Input of historical hourly electricity readings.
- Next-hour electricity consumption prediction.
- Recent 24-hour average consumption.
- Visual representation of historical readings.
- Rule-based energy-saving suggestions.
- Model information and prediction results.
- Input validation and error handling.
- Demo data generation for testing.

## 8. API Endpoints

The backend is developed using FastAPI.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the web application |
| GET | `/health` | Checks API health |
| POST | `/predict` | Generates electricity consumption prediction |

### Prediction Input

The `/predict` endpoint accepts:

- `history`: At least 168 non-negative hourly consumption readings.
- `last_timestamp`: Timestamp of the latest reading.

The 168 readings represent seven days of hourly historical data.

### Prediction Output

The response includes:

- `predicted_consumption_kw`
- `recent_24_hour_average_kw`
- `model_used`
- `prediction_for`
- `suggestions`

Invalid inputs are rejected using HTTP 422 validation responses.

## 9. Installation and Setup

### Prerequisites

- Python 3.11 or compatible Python version
- Git
- Docker (optional)

### Clone the Repository

```bash
git clone https://github.com/saksharr/wattwizz.git
cd wattwizz
```

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python train.py
```

### Run the Application

```bash
uvicorn app:app --reload --port 8000
```

Open the application:

http://127.0.0.1:8000

## 10. Running MLflow

Start the MLflow tracking dashboard:

```bash
python3 -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open:

http://127.0.0.1:5000

The dashboard displays the tracked experiments, parameters and evaluation metrics.

## 11. Running Tests

Run the automated test suite:

```bash
pytest -v
```

The test suite covers:

- Health endpoint.
- Homepage availability.
- Valid prediction requests.
- Insufficient historical readings.
- Negative consumption values.

## 12. Docker Setup

Build the Docker image:

```bash
docker build -t wattwizz .
```

Run the container:

```bash
docker run -d -p 8000:8000 --restart unless-stopped wattwizz
```

Access the application:

http://localhost:8000

The application uses a lightweight Python slim Docker base image and serving-specific dependencies.

## 13. Deployment

WattWizz is deployed using Render.

**Live URL:** https://wattwizz-p2y7.onrender.com

The application runs inside a Docker container and exposes the FastAPI backend and frontend through a public web service.

```

## 14. Limitations and Future Improvements

### Limitations

- Prediction accuracy depends on the quality and quantity of historical readings.
- The model may not capture sudden changes in household consumption.
- Energy-saving suggestions are rule-based rather than generated by a separate machine learning model.
- The current application uses historical input readings supplied by the user.

### Future Improvements

- Integration with real-time smart electricity meters.
- Longer-term electricity consumption forecasting.
- Improved model performance through hyperparameter tuning.
- More personalized energy-saving recommendations.
- Automated model retraining and drift monitoring.

## 16. Conclusion

WattWizz demonstrates how machine learning, time-series feature engineering and web application development can be combined to build a practical electricity consumption prediction system.

The project includes data preprocessing, model comparison, experiment tracking, model deployment, API development, automated testing and Docker-based deployment.

The application aims to help users better understand household electricity consumption patterns and make informed energy usage decisions.

---

**Project:** WattWizz – Electricity Consumption Prediction

**GitHub:** https://github.com/saksharr/wattwizz

**Live Application:** https://wattwizz-p2y7.onrender.com
