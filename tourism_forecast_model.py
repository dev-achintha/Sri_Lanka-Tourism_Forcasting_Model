import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class TourismForecastModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.event_impact = {
            "low": 0.9,
            "medium": 0.7,
            "high": 0.5,
            "severe": 0.3
        }

    def prepare_data(self, data, for_training=True):
        required_columns = ['Year', 'Month', 'Arrivals']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        data = data.copy()

        try:
            data['Date'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Month'], format='%Y-%B')
        except Exception as e:
            raise ValueError(f"Error converting 'Year' and 'Month' to datetime: {str(e)}")
        
        data = data.sort_values('Date')

        for i in range(1, 13):
            data[f'Arrivals_Lag_{i}'] = data['Arrivals'].shift(i)

        data['Month_Num'] = data['Date'].dt.month
        data['Year_Num'] = data['Date'].dt.year

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if for_training:
            data = data.dropna()
        else:
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

        return data

    def train(self, data):
        try:
            prepared_data = self.prepare_data(data, for_training=True)

            features = [col for col in prepared_data.columns if col.startswith('Arrivals_Lag_') or col in ['Month_Num', 'Year_Num']]
            X = prepared_data[features]
            y = prepared_data['Arrivals']

            X_scaled = self.scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"Model trained. MSE: {mse}, R2 Score: {r2}")
        
        except Exception as e:
            print(f"Error during training: {str(e)}")

    def predict(self, input_data, event_impact=None):
        try:
            prepared_data = self.prepare_data(input_data, for_training=False)

            features = [col for col in prepared_data.columns if col.startswith('Arrivals_Lag_') or col in ['Month_Num', 'Year_Num']]
            X = prepared_data[features].iloc[-1:]
            X_scaled = self.scaler.transform(X)

            prediction = self.model.predict(X_scaled)[0]


            if event_impact:
                prediction *= self.event_impact.get(event_impact, 1)

            return prediction
        
        except Exception as e:
            raise ValueError(f"Error during prediction: {str(e)}")

    def save_model(self, filename='tourism_forecast_model.joblib'):
        try:
            joblib.dump((self.model, self.scaler), filename)
            print(f"Model saved to {filename}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")

    def load_model(self, filename='tourism_forecast_model.joblib'):
        try:
            self.model, self.scaler = joblib.load(filename)
            print(f"Model loaded from {filename}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
