from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load models once when the server starts
pm25_model = joblib.load("models/rf_model_pm2.5.pkl")
pm10_model = joblib.load("models/rf_model_pm10.pkl")

# Define FastAPI app
app = FastAPI()

# Define request body schema
class InputData(BaseModel):
    aod: float
    HFLUX: float
    PBLH: float
    QLML: float
    SPEED: float
    TLML: float

@app.post("/predict")
def predict_pm(input: InputData):
    # Step 1: Create input features for PM2.5
    features_pm25 = pd.DataFrame([{
        "aod": input.aod,
        "HFLUX": input.HFLUX,
        "PBLH": input.PBLH,
        "QLML": input.QLML,
        "SPEED": input.SPEED,
        "TLML": input.TLML
    }])

    # Step 2: Predict PM2.5
    pm25_pred = pm25_model.predict(features_pm25)[0]

    # Step 3: Insert PM2.5 as a feature for PM10 (after 'aod')
    features_pm25.insert(loc=features_pm25.columns.get_loc('aod') + 1, column='PM2.5', value=pm25_pred)

    # Step 4: Predict PM10
    pm10_pred = pm10_model.predict(features_pm25)[0]

    # Return results
    return {
        "Predicted_PM2.5": round(pm25_pred, 2),
        "Predicted_PM10": round(pm10_pred, 2)
    }
