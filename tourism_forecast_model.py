import pandas as pd
from prophet import Prophet
import joblib

class TourismForecastModel:
    def __init__(self):
        self.prophet_model = None

    def prepare_data(self, data):
        required_columns = ['Year', 'Month', 'Arrivals']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data = data.copy()
        data['ds'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%B')
        data.rename(columns={'Arrivals': 'y'}, inplace=True)
        return data[['ds', 'y']]

    def train(self, data):
        try:
            prepared_data = self.prepare_data(data)
            self.prophet_model = Prophet()
            self.prophet_model.fit(prepared_data)
            print("Prophet model trained.")
        except Exception as e:
            print(f"Error during training: {str(e)}")

    def predict(self, future_periods):
        try:
            future = self.prophet_model.make_future_dataframe(periods=future_periods, freq='M')
            forecast = self.prophet_model.predict(future)
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        except Exception as e:
            print(f"Prediction Error: {str(e)}")  # Print the specific error
            raise ValueError(f"Error during prediction: {str(e)}")

    def save_model(self, filename='tourism_forecast_model.joblib'):
        try:
            # Only save the Prophet model
            joblib.dump(self.prophet_model, filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filename='tourism_forecast_model.joblib'):
        try:
            # Load only the Prophet model
            self.prophet_model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
