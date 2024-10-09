# In Progress
# Sri Lanka's Tourism Forecasting Model

## [Go to the Notebook ◳](https://github.com/dev-achintha/Sri_Lanka-Tourism_Forcasting_Model/blob/prophet/notebooks/Sri_Lankan_TourismForecastModel_Prophet.ipynb)

## 1. Setup and Data Preparation

```python
import pandas as pd
df = pd.read_csv('../dataset/2015-2024-monthly-tourist-arrivals-sl-csv.csv')
df = df.drop(columns=['PercentageChange'], errors='ignore')
```

We load the tourism dataset (2014-August 2024) from [SLTDA](https://www.sltda.gov.lk/en/monthly-tourist-arrivals-reports). The 'PercentageChange' column is removed as it’s irrelevant to absolute arrival forecasting.

## 2. Date Formatting

```python
df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
df.rename(columns={'Arrivals': 'y'}, inplace=True)
df = df[['ds', 'y']]
```

We prepare the data for Prophet:
1. Convert 'Year' and 'Month' into a date column 'ds'.
2. Rename 'Arrivals' to 'y' as required by Prophet.

## 3. Model Creation and Configuration

```python
from prophet import Prophet
model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, changepoint_prior_scale=0.1, n_changepoints=30)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
```

We use Prophet for its robust handling of trend shifts and seasonality:
1. Yearly and weekly seasonality are included, but daily is ignored (monthly data).
2. Changepoints and trend flexibility are controlled by `changepoint_prior_scale=0.1` and `n_changepoints=30`.
3. Custom monthly seasonality is added to better capture tourism cycles.

## 4. Model Fitting

```python
model.fit(df)
```

Prophet fits the data, identifying the trend, seasonality, and changepoints using Bayesian methods.

## 5. Future Predictions

```python
future_periods = 24
future = model.make_future_dataframe(periods=future_periods, freq='M')
forecast = model.predict(future)
```

We forecast 24 months ahead using Prophet's future dataframe generation. The result includes predictions (`yhat`) and uncertainty intervals (`yhat_lower`, `yhat_upper`).

## 6. Visualization

```python
import matplotlib.pyplot as plt

fig = model.plot(forecast)
plt.title('Tourism Forecast')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['ds'], df['y'], label='Actual Arrivals')
ax.plot(forecast['ds'], forecast['yhat'], label='Predicted Arrivals')
plt.show()
```

Two plots: one using Prophet's built-in functionality and another custom plot comparing actual vs predicted values.

## 7. Model Evaluation

```python
from sklearn.metrics import mean_absolute_error

actual = df['y']
predicted = forecast.loc[forecast['ds'].isin(df['ds']), 'yhat']
mae = mean_absolute_error(actual, predicted)
print(f"Mean Absolute Error: {mae}")
```

We calculate Mean Absolute Error (MAE) to evaluate the model's performance. The high MAE and observed forecast issues suggest potential overfitting and missed external factors affecting tourism.

Future improvements include tuning changepoint parameters.
