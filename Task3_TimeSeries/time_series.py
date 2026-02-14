import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# =============================
# 1️⃣ Load Dataset
# =============================
df = pd.read_csv(
    "airline-passengers.csv",
    parse_dates=["Month"],
    index_col="Month"
)

print("First rows:\n", df.head())

# =============================
# 2️⃣ Original Trend Plot
# =============================
plt.figure(figsize=(10,5))
plt.plot(df['Passengers'])
plt.title("Monthly Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers")
plt.show()

# =============================
# 3️⃣ Decomposition (Trend + Seasonality)
# =============================
decomposition = seasonal_decompose(df['Passengers'], model='multiplicative')

decomposition.plot()
plt.show()

# =============================
# 4️⃣ Moving Averages
# =============================
df['MA_6'] = df['Passengers'].rolling(window=6).mean()
df['MA_12'] = df['Passengers'].rolling(window=12).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Passengers'], label='Actual')
plt.plot(df['MA_6'], '--', label='6-Month MA')
plt.plot(df['MA_12'], label='12-Month MA')
plt.legend()
plt.title("Moving Average Trend")
plt.show()

# =============================
# 5️⃣ Forecast using ARIMA
# =============================
train = df.iloc[:-12]
test = df.iloc[-12:]

model = ARIMA(train['Passengers'],
              order=(2,1,1),
              seasonal_order=(1,1,1,12))

model_fit = model.fit()

forecast = model_fit.forecast(steps=12)

# Error measurement
rmse = np.sqrt(mean_squared_error(test['Passengers'], forecast))
print("RMSE:", rmse)

# Forecast Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train['Passengers'], label='Train')
plt.plot(test.index, test['Passengers'], label='Actual')
plt.plot(test.index, forecast, '--', label='Forecast')
plt.legend()
plt.title("Passenger Forecast")
plt.show()

print("\nTime Series Analysis Completed ✅")

# keeps window open
input("Press Enter to exit...")