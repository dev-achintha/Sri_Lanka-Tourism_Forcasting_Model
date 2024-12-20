import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go

st.sidebar.title("Tourism Forecasting with Custom Models")
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model", 
    ["Custom Prophet", "Custom SARIMA", "Custom LSTM"]
)

uploaded_file = st.sidebar.file_uploader("Upload new data (CSV)", type="csv")

# Load Data
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(
        "https://raw.githubusercontent.com/dev-achintha/Sri_Lanka-Tourism_Forcasting_Model/working/dataset/2015-2024-monthly-tourist-arrivals-sl-csv.csv"
    )

# Preprocess Data
df['ds'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'], format='%Y-%B')
df = df[['ds', 'Arrivals']].rename(columns={'Arrivals': 'y'}).sort_values('ds')

st.sidebar.subheader("Changepoint Management")
if 'changepoints' not in st.session_state:
    st.session_state['changepoints'] = []

changepoints_df = pd.DataFrame(st.session_state['changepoints'], columns=['Date', 'Impact Level'])
st.sidebar.table(changepoints_df)

# Add Changepoint
new_date = st.sidebar.date_input("Select Changepoint Date")
impact_level = st.sidebar.slider("Impact Level", -100, 100, 0)
if st.sidebar.button("Add Changepoint"):
    st.session_state['changepoints'].append({'Date': new_date, 'Impact Level': impact_level})
    st.session_state['trigger_rerun'] = not st.session_state.get('trigger_rerun', False)  # Trigger refresh

# Models

def custom_prophet(data, periods=12):
    """Custom Prophet Model for Forecasting."""
    model = Prophet(yearly_seasonality=True)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def custom_sarima(data, periods=12):
    """Custom SARIMA Model for Forecasting."""
    model = SARIMAX(data['y'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=periods)
    
    # Extract predicted values
    forecast_df = forecast.predicted_mean.reset_index()  
    forecast_df.columns = ['ds', 'yhat']
    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
    
    return forecast_df

def custom_lstm(data, periods=12):
    """Custom LSTM Model for Forecasting."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['y']])

    # Create sequences
    seq_length = 12
    X_train, y_train = [], []
    for i in range(len(scaled_data) - seq_length):
        X_train.append(scaled_data[i:i + seq_length])
        y_train.append(scaled_data[i + seq_length])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # LSTM Model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    # Generate forecast
    forecast = []
    input_seq = scaled_data[-seq_length:]
    for _ in range(periods):
        input_seq = input_seq.reshape((1, seq_length, 1))
        y_pred = model.predict(input_seq, verbose=0)[0][0]
        forecast.append(y_pred)
        input_seq = np.append(input_seq[0][1:], y_pred).reshape(-1, 1)

    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    forecast_df = pd.DataFrame({'ds': pd.date_range(data['ds'].max(), periods=periods, freq='M'), 'yhat': forecast})
    return forecast_df

# Select Model
if model_choice == "Custom Prophet":
    forecast = custom_prophet(df)
elif model_choice == "Custom SARIMA":
    forecast = custom_sarima(df)
else:
    forecast = custom_lstm(df)

# Plot Forecast
st.subheader(f"{model_choice} Forecast")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=forecast['ds'], y=forecast['yhat'], 
    mode='lines', name='Forecast', line=dict(color='yellow')
))

fig.add_trace(go.Scatter(
    x=df['ds'], y=df['y'], 
    mode='markers', name='Actual Data', marker=dict(color='blue')
))

fig.update_layout(
    title=f"{model_choice} Forecast", 
    xaxis_title='Date', 
    yaxis_title='Arrivals',
    xaxis=dict(tickformat='%Y')
)

st.plotly_chart(fig)

# Export Forecast Data
st.download_button(
    label="Download Forecast",
    data=forecast.to_csv(index=False).encode('utf-8'),
    file_name=f'{model_choice}_forecast.csv',
    mime='text/csv'
)
