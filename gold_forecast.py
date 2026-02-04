import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet

# Download gold price data (GC=F)
data = yf.download("GC=F", start="2016-01-01")

# Keep only Date and Close
df = data[['Close']].reset_index()
df.columns = ['ds', 'y']   # Prophet requires ds (date) and y (value)

print(df.head())

# Plot original data
plt.figure(figsize=(10,5))
plt.plot(df['ds'], df['y'])
plt.title("Gold Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()

# Build Prophet model
model = Prophet()
model.fit(df)

# Create future dates (next 365 days)
future = model.make_future_dataframe(periods=365)

# Forecast
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Gold Price Forecast")
plt.show()

# Plot components (trend, yearly seasonality)
model.plot_components(forecast)
plt.show()
