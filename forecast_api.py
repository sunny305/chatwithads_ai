from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from mlforecast import MLForecast
from sklearn.linear_model import Ridge
from datetime import datetime
import requests



# ------------------------
# Insight Prompt Formatter
# ------------------------
def format_forecast_for_prompt(forecast_data):
    forecast_data = sorted(forecast_data, key=lambda x: x["ds"])
    text = "Here's the predicted campaign performance for the next few days:\n"
    for row in forecast_data:
        ds = row["ds"]
        # Ensure date is a datetime object
        if not isinstance(ds, datetime):
            ds = pd.to_datetime(ds)
        value = round(row["Ridge"], 2)
        text += f"- {ds.strftime('%B %d, %Y')}: {value}\n"
    text += "\nGenerate a brief, clear marketing insight from this forecast. Focus on trends or changes."
    return text

def generate_insight_with_mistral(prompt: str):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json().get("response", "").strip()

# ------------------------
# Request Models
# ------------------------
class TimeSeriesItem(BaseModel):
    unique_id: str
    ds: str  # date
    y: float

class ForecastRequest(BaseModel):
    data: List[TimeSeriesItem]
    days: int  # How many days to forecast


# ------------------------
# FastAPI App
# ------------------------
app = FastAPI()


# ------------------------
# /forecast Endpoint
# ------------------------
@app.post("/forecast")
def forecast_endpoint(request: ForecastRequest):
    # Convert incoming data to DataFrame
    df = pd.DataFrame([item.dict() for item in request.data])
    df['ds'] = pd.to_datetime(df['ds'])

    # Forecast setup
    fcst = MLForecast(
        models=[Ridge()],
        lags=[1, 2],
        freq='D'
    )
    fcst.fit(df)
    forecast_df = fcst.predict(request.days)

    # Convert to records + generate prompt
    forecast_records = forecast_df.to_dict(orient="records")
    prompt = format_forecast_for_prompt(forecast_records)
    insight = generate_insight_with_mistral(prompt)


    return {
        "forecast": forecast_records,
        "prompt": prompt,
        "insight": insight
    }
