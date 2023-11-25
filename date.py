import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Set the default 'From' date to '2022-01-01'
default_from_date = datetime(2022, 1, 1)
st.set_page_config(
    page_title="Date Range Selector",
    page_icon="✅",
    layout="wide"
)

# Add custom CSS for background image
st.markdown(
    """
    <style>
        body {
            background-image: url('your_background_image_url.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title('STOCK MARKET FORECASTING USING TIME SERIES ANALYSIS')

# Date input fields
from_date = st.date_input('From Date', value=default_from_date)
to_date = st.date_input('To Date')

# Ensure 'From' date is not after 'To' date
if from_date > to_date:
    st.error('Please select a valid date range. The "From" date should be before the "To" date.')
else:
    st.success('Valid date range selected.')

    # Generate button
    if st.button('Generate'):
        # Use a spinner to indicate processing
        with st.spinner("Generating data. Please wait... It may take some time."):
            # Calculate the difference in days and store it in the variable 'forsteps'
            forsteps = (to_date - from_date).days
            ticker = 'TSLA'
            start_date = '2015-01-01'
            end_date = '2022-01-01'
            tesla_data = yf.download(ticker, start=start_date, end=end_date)
            tesla_data.reset_index(inplace=True)
            tesla_data.isnull().sum()
            tesla_data = tesla_data[['Date', 'Close']]

            # Convert 'Date' to datetime and set it as the index
            tesla_data['Date'] = pd.to_datetime(tesla_data['Date'])
            tesla_data.set_index('Date', inplace=True)

            # Load the pre-trained LSTM model using keras.models.load_model
            lstm_model = keras.models.load_model('model.h5')  # Replace with the actual path

            # Assuming 'tesla_data' is your dataframe containing the stock data
            # Extract the data for the prediction period (from 2022-01-02 to 2023-01-01)
            prediction_start_date = '2021-10-01'
            prediction_end_date = '2022-01-01'
            prediction_data = tesla_data[(tesla_data.index >= prediction_start_date) & (tesla_data.index <= prediction_end_date)]

            # Check if there is enough data for prediction
            if len(prediction_data) < 60:
                st.warning("Insufficient data for prediction. Please ensure at least 60 days of historical data.")
            else:
                # Extract the last 60 days' closing prices as input for prediction
                input_data = prediction_data['Close'].values[-60:].reshape(1, -1, 1)

                # Normalize the input data
                scaler = MinMaxScaler(feature_range=(0, 1))
                input_data = scaler.fit_transform(input_data.reshape(-1, 1)).reshape(1, -1, 1)

                # Make predictions for future values
                forecast_steps = forsteps  # Adjust the number of steps into the future
                forecast = []

                for _ in range(forecast_steps):
                    next_value = lstm_model.predict(input_data)
                    forecast.append(next_value[0, 0])
                    input_data = np.append(input_data, [[[next_value[0, 0]]]], axis=1)

                # Denormalize the forecasted values
                forecast = np.array(forecast).reshape(-1, 1)
                forecast = scaler.inverse_transform(forecast).flatten()

                # Create a date range for the forecasted values
                forecast_dates = pd.date_range(start=from_date, periods=forecast_steps + 1, freq='D')[1:]
                forecast_dates = forecast_dates.date
                forecast_data = pd.DataFrame({'Date': forecast_dates, 'Close': forecast})

                # Create a layout with two columns using st.columns
                col1, col2 = st.columns([2, 1])  # Adjust the width ratio as needed

                # Increase the size of the graph
                # plt.figure(figsize=(15, 10))

                # # Plot the original time series and the forecast in the first column
                # with col1:
                #     # plt.figure(figsize=(15, 6))
                #     plt.plot(tesla_data.index, tesla_data['Close'], label='Original Time Series')
                #     plt.plot(prediction_data.index, prediction_data['Close'], label='Historical Data', color='gray')
                #     plt.plot(forecast_dates, forecast, label='Forecast')
                #     plt.xlabel('Date')
                #     plt.ylabel('Close Price')
                #     plt.legend(fontsize=24)
                #     plt.title('LSTM Stock Price Forecasting')
                #     plt.xticks(fontsize=24)  # Increase x-axis tick font size
                #     plt.yticks(fontsize=24)  # Increase y-axis tick font size
                #     plt.title('LSTM Stock Price Forecasting', fontsize=32)
                #     st.pyplot(plt)

                # # Display the forecast DataFrame in the second column
                # with col2:
                #     st.dataframe(forecast_data)

                fig = go.Figure()

                # Plot the original time series and the forecast in the first column
                fig.add_trace(go.Scatter(x=tesla_data.index, y=tesla_data['Close'], mode='lines', name='Original Time Series'))
               # fig.add_trace(go.Scatter(x=prediction_data.index, y=prediction_data['Close'], mode='lines', name='Historical Data', line=dict(color='red')))
                fig.add_trace(go.Scatter(x=forecast_dates, y=forecast, mode='lines', name='Forecast',line=dict(color='red')))

                # Set layout properties
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    title='LSTM Stock Price Forecasting',
                    legend=dict(font=dict(size=14)),  # Increase legend font size
                    font=dict(size=14),  # Increase overall font size
                )

                # Display the graph in the first column
                with col1:
                    st.plotly_chart(fig)

                # Display the forecast DataFrame in the second column
                with col2:
                    st.dataframe(forecast_data)
