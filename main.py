from fastapi import FastAPI, HTTPException, Form, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
from tourism_forecast_model import TourismForecastModel
import os
import joblib
from datetime import datetime

app = FastAPI()

model = TourismForecastModel()

# Path to the dataset and model file
model_file_path = 'model/tourism_forecast_model.joblib'
dataset_path = 'dataset/2015-2024-monthly-tourist-arrivals-sl-csv.csv'

# Ensure the model is trained or loaded when the app starts
def train_model_on_startup():
    try:
        if os.path.exists(model_file_path):
            # Load the previously trained model
            model.load_model(model_file_path)
            print(f"Model loaded from {model_file_path}")
        else:
            # If no model exists, train the model and save it
            df = pd.read_csv(dataset_path)
            model.train(df)
            model.save_model(model_file_path)
            print("Model trained and saved successfully on startup.")
    except Exception as e:
        print(f"Error during model loading or training on startup: {str(e)}")

# Call the train function when the app starts
train_model_on_startup()

class PredictionInput(BaseModel):
    year: int = Field(..., example=2024, description="Year for the prediction")
    month: Optional[str] = Field(None, example="August", description="Month for the prediction (optional)")
    event_impact: Optional[str] = Field(None, example="medium", description="Event impact severity (optional: low, medium, high, severe)")

@app.post("/predict", summary="Predict Tourist Arrivals", description="Predict the number of tourist arrivals for a specific month and year.")
async def predict(
    year: int = Form(..., description="Year for the prediction"),
    month: Optional[str] = Form(None, description="Month for the prediction (optional)"),
    event_impact: Optional[str] = Form(None, description="Event impact severity (optional: low, medium, high, severe)")
):
    try:
        # Load the existing dataset
        df = pd.read_csv(dataset_path)

        # Prepare input for prediction
        df_input = pd.DataFrame([{
            'Year': year,
            'Month': month if month else "January",  # Default to January if no month is provided
            'Arrivals': None  # Placeholder for prediction
        }])

        # Add the last 12 months of data for the model to predict based on it
        last_12_months = df.tail(12)
        df_input = pd.concat([last_12_months, df_input], ignore_index=True)

        # Make the prediction
        prediction = model.predict(df_input, event_impact)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain", summary="Retrain the model with new data")
async def retrain(
    file: UploadFile = File(..., description="Upload a CSV file with 'year', 'month', and 'arrivals' columns"),
    combine_with_existing: bool = Form(True, description="Whether to combine with the existing dataset")
):
    try:
        # Load the new data from the uploaded file
        new_data = pd.read_csv(file.file)

        # Ensure the required columns are present
        required_columns = ['year', 'month', 'arrivals']
        if not all(col in new_data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="CSV file must include 'year', 'month', and 'arrivals' columns.")

        # Save the new dataset with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        new_dataset_path = f"dataset/new_dataset_{timestamp}.csv"
        new_data.to_csv(new_dataset_path, index=False)

        # Combine with the existing dataset if requested
        if combine_with_existing:
            existing_data = pd.read_csv(dataset_path)
            combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        else:
            combined_data = new_data

        # Retrain the model with the combined data
        model.train(combined_data)
        model.save_model(model_file_path)  # Save the updated model
        return {"message": "Model retrained and saved successfully", "new_dataset": new_dataset_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
