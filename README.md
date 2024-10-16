# Sri Lankan Tourism Forecasting Model

**Date:** October 16, 2024  (Last Update)

**Author:** [Achintha](achintha.me)

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Methodology](#methodology)
  - [1. Data Loading and Preprocessing](#1-data-loading-and-preprocessing)
  - [2. Data Splitting](#2-data-splitting)
  - [3. Forecasting Models](#3-forecasting-models)
    - [a. Prophet](#a-prophet)
    - [b. SARIMA](#b-sarima)
    - [c. LSTM](#c-lstm)
    - [d. Exponential Smoothing](#d-exponential-smoothing)
  - [4. Generating 12-Month Forecasts](#4-generating-12-month-forecasts)
  - [5. Visualization](#5-visualization)
  - [6. Model Evaluation](#6-model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## Introduction

The tourism industry is a significant contributor to Sri Lanka's economy. Accurate forecasting of tourist arrivals helps policymakers, businesses, and stakeholders make informed decisions to enhance the visitor experience, optimize resource allocation, and plan for future growth. This project leverages multiple time series forecasting models to predict tourist arrivals in Sri Lanka for the upcoming 12 months.

## Project Overview

This project implements four distinct forecasting models to predict monthly tourist arrivals in Sri Lanka:

1. **Prophet**: An additive model by Facebook designed for forecasting time series data.
2. **SARIMA** (Seasonal Autoregressive Integrated Moving Average): A statistical model that accounts for seasonality.
3. **LSTM** (Long Short-Term Memory): A type of recurrent neural network suitable for sequence prediction.
4. **Exponential Smoothing**: A time series forecasting method that applies weighted averages.

Each model is trained on historical data, forecasts the next 12 months, and its performance is evaluated against actual data using the Mean Absolute Error (MAE) metric.

## Data Source

The dataset used in this project is sourced from a GitHub repository:
- **URL**: [Sri Lanka Monthly Tourist Arrivals (2014-2024)](https://github.com/dev-achintha/Sri_Lanka-Tourism_Forcasting_Model/prophet/dataset/2014-2024-monthly-tourist-arrivals-sl-csv.csv)

**Dataset Description:**
- **Year**: The year of the record.
- **Month**: The month of the record.
- **Arrivals**: Number of tourist arrivals in that month.
- **PercentageChange**: Monthly percentage change in arrivals (dropped in preprocessing).

## Methodology

### 1. Data Loading and Preprocessing

**Objective:** Prepare the raw data for modeling by handling missing values, formatting dates, and selecting relevant features.

**Steps:**
- **Data Import**: The dataset is loaded directly from a GitHub repository using `pandas`.
- **Data Cleaning**: The `PercentageChange` column is dropped as it's not required for forecasting.
- **Date Formatting**: A new datetime column `ds` is created by combining `Year` and `Month`. Prophet requires a datetime column named `ds` and a target variable named `y`.
- **Renaming Columns**: `Arrivals` is renamed to `y` to align with Prophet's expectations.
- **Data Sorting**: The data is sorted chronologically to ensure temporal consistency.

**Code Snippet:**
```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dev-achintha/Sri_Lanka-Tourism_Forcasting_Model/prophet/dataset/2014-2024-monthly-tourist-arrivals-sl-csv.csv')
df = df.drop(columns=['PercentageChange'], errors='ignore')
df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
df.rename(columns={'Arrivals': 'y'}, inplace=True)
df = df[['ds', 'y']]
df = df.sort_values('ds')
```

### 2. Data Splitting

**Objective:** Divide the dataset into training and testing sets to evaluate model performance on unseen data.

**Steps:**
- The last 12 months of data are reserved for testing, providing a robust evaluation period.
- The remaining data serves as the training set for model fitting.

**Code Snippet:**
```python
# Split data into training and testing sets (last 12 months for testing)
train = df[:-12]
test = df[-12:]
```

### 3. Forecasting Models

Four distinct models are implemented, each with unique characteristics and strengths.

#### a. Prophet

**Overview:**
Prophet is an additive regression model with built-in seasonality, designed to handle time series data with multiple seasonalities and trend changes.

**Configuration:**
- **Yearly Seasonality**: Captures annual patterns.
- **Monthly Seasonality**: Added manually to capture monthly fluctuations.
- **Changepoints**: Points where the trend changes. Custom changepoints can be added based on known events.

**Code Snippet:**
```python
from prophet import Prophet

model_prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.1,
    n_changepoints=30
)
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_prophet.fit(train)
```

#### b. SARIMA

**Overview:**
SARIMA extends the ARIMA model by adding seasonal terms, making it suitable for data with seasonal patterns.

**Configuration:**
- **Order**: `(1, 1, 1)` indicating autoregressive, differencing, and moving average components.
- **Seasonal Order**: `(1, 1, 1, 12)` capturing seasonal dependencies with a 12-month cycle.

**Code Snippet:**
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

order = (1, 1, 1)
seasonal_order = (1, 1, 1, 12)
model_sarima = SARIMAX(train['y'], order=order, seasonal_order=seasonal_order)
results_sarima = model_sarima.fit(disp=False)
```

#### c. LSTM

**Overview:**
LSTM networks are a type of recurrent neural network capable of learning long-term dependencies, making them effective for sequence prediction tasks.

**Configuration:**
- **Data Scaling**: Features are scaled between 0 and 1 for efficient training.
- **Sequence Creation**: Sequences of 12 months are used as input to predict the next month.
- **Model Architecture**: Consists of an LSTM layer followed by a Dense layer.
- **Training**: The model is trained over 100 epochs with a batch size of 32.

**Code Snippet:**
```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train[['y']])

# Creating sequences
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

seq_length = 12
X, y_lstm = create_sequences(scaled_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Building the model
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')

# Training the model
model_lstm.fit(X, y_lstm, epochs=100, batch_size=32, verbose=0)
```

#### d. Exponential Smoothing

**Overview:**
Exponential Smoothing forecasts future points by applying exponentially decreasing weights to past observations, capturing trends and seasonality.

**Configuration:**
- **Seasonal Periods**: 12 months to account for annual seasonality.
- **Trend and Seasonality**: Both modeled as additive components.

**Code Snippet:**
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Building and fitting the model
model_es = ExponentialSmoothing(
    train['y'],
    seasonal_periods=12,
    trend='add',
    seasonal='add'
)
results_es = model_es.fit(optimized=True, remove_bias=True)
```

### 4. Generating 12-Month Forecasts

Each model generates forecasts for the next 12 months following the training period.

#### Prophet Forecasting
```python
# Creating future dates for 12 months
future_prophet = model_prophet.make_future_dataframe(periods=12, freq='MS')
forecast_prophet = model_prophet.predict(future_prophet)
forecast_prophet = forecast_prophet[['ds', 'yhat']].tail(12)
```

#### SARIMA Forecasting
```python
# Forecasting next 12 months using SARIMA
forecast_sarima = results_sarima.get_forecast(steps=12)
forecast_sarima = forecast_sarima.predicted_mean.reset_index(drop=True)
```

#### LSTM Forecasting
```python
# Initializing the last sequence from training data
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
forecast_lstm_scaled = []

# Iteratively predicting each month
for _ in range(12):
    next_pred = model_lstm.predict(last_sequence, verbose=0)
    forecast_lstm_scaled.append(next_pred[0, 0])
    # Updating the sequence with the latest prediction
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[0, -1, 0] = next_pred

# Inverse scaling to obtain actual values
forecast_lstm = scaler.inverse_transform(np.array(forecast_lstm_scaled).reshape(-1, 1)).flatten()
```

#### Exponential Smoothing Forecasting
```python
# Forecasting next 12 months using Exponential Smoothing
forecast_es = results_es.forecast(12).reset_index(drop=True)
```

### 5. Visualization

**Objective:** Compare actual tourist arrivals with forecasts from all models to visually assess performance.

**Steps:**
- **Plot Setup**: A single plot displays actual data and forecasts from each model.
- **Colors and Labels**: Different colors and labels distinguish each model's forecast.
- **Enhancements**: Titles, axis labels, legends, gridlines, and layout adjustments improve readability.

**Code Snippet:**
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
plt.plot(df['ds'], df['y'], label='Actual', color='black')

# Plot Prophet Forecast
plt.plot(forecast_prophet['ds'], forecast_prophet['yhat'], label='Prophet', color='blue')

# Plot SARIMA Forecast
plt.plot(test['ds'], forecast_sarima, label='SARIMA', color='red')

# Plot LSTM Forecast
plt.plot(test['ds'], forecast_lstm, label='LSTM', color='green')

# Plot Exponential Smoothing Forecast
plt.plot(test['ds'], forecast_es, label='Exponential Smoothing', color='orange')

plt.title('Actual vs Predicted Tourist Arrivals')
plt.xlabel('Date')
plt.ylabel('Number of Arrivals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

**Sample Plot:**

![Actual vs Predicted Tourist Arrivals](https://imgur.com/a/V2NBGPy)

### 6. Model Evaluation

**Objective:** Quantify the accuracy of each forecasting model using the Mean Absolute Error (MAE) metric.

**Steps:**
- **Calculate MAE**: For each model, compare forecasted values against actual test data.
- **Display Results**: Print MAE values to identify the best-performing model.

**Code Snippet:**
```python
from sklearn.metrics import mean_absolute_error

mae_prophet = mean_absolute_error(test['y'], forecast_prophet['yhat'])
mae_sarima = mean_absolute_error(test['y'], forecast_sarima)
mae_lstm = mean_absolute_error(test['y'], forecast_lstm)
mae_es = mean_absolute_error(test['y'], forecast_es)

print(f"Mean Absolute Error (MAE) for Each Model:")
print(f"Prophet: {mae_prophet:.2f}")
print(f"SARIMA: {mae_sarima:.2f}")
print(f"LSTM: {mae_lstm:.2f}")
print(f"Exponential Smoothing: {mae_es:.2f}")
```

**Sample Output:**
```
Mean Absolute Error (MAE) for Each Model:
Prophet: 87229.15
SARIMA: 68969.09
LSTM: 93762.59
Exponential Smoothing: 83541.29
```

## Results

The model evaluation reveals the following MAE values:

- **SARIMA**: 68,969.09
- **Exponential Smoothing**: 83,541.29
- **Prophet**: 87,229.15
- **LSTM**: 93,762.59

**Interpretation:**
- **SARIMA** outperforms the other models with the lowest MAE, indicating it provides the most accurate forecasts among the implemented models for this dataset.
- **Exponential Smoothing** follows, offering better performance than Prophet and LSTM.
- **Prophet** and **LSTM** exhibit higher MAE values, suggesting room for improvement through hyperparameter tuning or alternative modeling approaches.

## Conclusion

This project successfully forecasts Sri Lankan tourist arrivals for the next 12 months using four different models. Among the models tested, SARIMA demonstrated superior performance based on the MAE metric. While LSTM offers potential for capturing complex patterns, its performance in this case was less optimal, possibly due to the dataset's size or the need for more extensive tuning.

**Key Takeaways:**
- **SARIMA** is a robust model for time series data with clear seasonal patterns.
- **Exponential Smoothing** serves as a reliable alternative, especially when SARIMA is not feasible.
- **Prophet** provides flexibility and is user-friendly for business forecasting but may require adjustment based on specific data characteristics.
- **LSTM** requires ample data and careful tuning to surpass traditional statistical models in accuracy.

## Dependencies

Ensure the following Python libraries are installed. You can install them using `pip`:

```bash
pip install pandas numpy prophet statsmodels scikit-learn keras tensorflow matplotlib
```

**Note:**  
- **Prophet** may require additional system dependencies. Refer to the [Prophet Installation Guide](https://facebook.github.io/prophet/docs/installation.html) for detailed instructions.
- **TensorFlow** and **Keras** versions should be compatible to avoid runtime issues.

## Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Sri_Lankan_TourismForecastModel.git
   cd Sri_Lankan_TourismForecastModel
   ```

2. **Install Dependencies:**
   As outlined in the [Dependencies](#dependencies) section.

3. **Run the Forecasting Script:**
   Ensure that you have a Python environment set up. Execute the script using:
   ```bash
   python forecast_tourism.py
   ```
   *Replace `forecast_tourism.py` with the actual script name.*

4. **View Results:**
   - **Plot**: A visualization comparing actual and predicted tourist arrivals.
   - **Console Output**: MAE values for each model to assess performance.

## Future Work

To enhance the forecasting accuracy and robustness of the models, I consider to implement the following:

1. **Hyperparameter Tuning:**
   - Optimize model parameters using grid search or other optimization techniques to improve performance.
  
2. **Cross-Validation:**
   - Implement time-series cross-validation to better assess model stability and generalizability.

3. **Feature Engineering:**
   - Incorporate additional features such as economic indicators, marketing campaigns, global events (e.g., pandemics, natural disasters), and exchange rates to enrich the models.

4. **Ensemble Methods:**
   - Combine forecasts from multiple models to potentially achieve better accuracy than individual models.

5. **Advanced Neural Networks:**
   - Explore more complex architectures like GRU or Transformers for capturing intricate temporal patterns.

6. **Incorporate External Data:**
   - Use data such as hotel bookings, flight inquiries, or search trends to provide external context to the models.

7. **User Interface Development:**
   - Develop a dashboard or web application to allow stakeholders to interact with the forecasting models and visualize results dynamically.

## Acknowledgments

- **Facebook Prophet**: [Prophet Documentation](https://facebook.github.io/prophet/)
- **Statsmodels**: [Statsmodels Documentation](https://www.statsmodels.org/)
- **Keras & TensorFlow**: [TensorFlow Documentation](https://www.tensorflow.org/)
- **Scikit-Learn**: [Scikit-Learn Documentation](https://scikit-learn.org/)
- **Matplotlib**: [Matplotlib Documentation](https://matplotlib.org/)
- **Data Source**: Provided by [SLTDA](https://www.sltda.gov.lk/en/statistics)