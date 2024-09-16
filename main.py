import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tourism_forecast_model import TourismForecastModel
import os

model = TourismForecastModel()

model_file_path = 'model/model.joblib'
dataset_path = 'dataset/2015-2024-monthly-tourist-arrivals-sl-csv.csv'

df = pd.read_csv(dataset_path)

def load_or_train_model():
    if os.path.exists(model_file_path):
        model.load_model(model_file_path)
        st.write(f"Model loaded from {model_file_path}")
    else:
        model.train(df)
        model.save_model(model_file_path)
        st.write("Prophet model trained and saved successfully.")

st.title("Sri Lanka Tourism Forecasting with Prophet")

load_or_train_model()

st.header("Long-term Forecast with Prophet")
future_months = st.number_input("Enter the number of months to forecast into the future:", min_value=1, value=12)

if st.button("Forecast"):
    forecast = model.predict(future_months)
    st.write(forecast.tail())

    # Plot forecast
    fig, ax = plt.subplots()
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
    ax.set_title(f"Prophet Forecast for {future_months} months")
    ax.set_xlabel("Date")
    ax.set_ylabel("Tourist Arrivals")
    ax.legend()
    st.pyplot(fig)

st.header("Retrain the Model")
uploaded_file = st.file_uploader("Upload a CSV file with 'Year', 'Month', and 'Arrivals' columns", type=["csv"])
combine_with_existing = st.checkbox("Combine with existing dataset?", value=True)

if st.button("Retrain"):
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        if not all(col in new_data.columns for col in ['Year', 'Month', 'Arrivals']):
            st.error("Uploaded CSV must contain 'Year', 'Month', and 'Arrivals' columns.")
        else:
            if combine_with_existing:
                combined_data = pd.concat([df, new_data], ignore_index=True)
            else:
                combined_data = new_data

            model.train(combined_data)
            model.save_model(model_file_path)
            st.success("Model retrained successfully.")
    else:
        st.error("Please upload a file to retrain the model.")

model.train(df)
model.save_model(model_file_path)
