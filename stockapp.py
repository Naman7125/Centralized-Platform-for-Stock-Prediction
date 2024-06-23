import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Define constants for the date range
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Set the title of the Streamlit app
st.title('Centralized Platform for Stock Prediction')

# Define the stock options
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# Define the prediction period in years
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

# Function to load data from Yahoo Finance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load the data and display a loading message
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

# Display the raw data
st.subheader('Raw data')
st.write(data.head())
st.write(data.tail())

# Function to plot the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Plot the raw data
plot_raw_data()

# Prepare the data for forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Create and fit the Prophet model
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display the forecast data and plots
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast) #forecast means predicted values
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

st.subheader('User Feedback')
feedback = st.text_area('Please provide your feedback or suggestions below:')
submit_button = st.button('Submit')

if submit_button:
    st.write('Thank you for your feedback!')
    # Here, you could add code to save the feedback to a file or database
    print(feedback)  # Printing feedback to the console (for demonstration)
    # Example: Save feedback to a file
    with open('user_feedback.txt', 'a') as f:
        f.write(f"{feedback}\n")