# -----------------------------------------------------------------------------
# Project 2: Sales Forecasting Model (Python, scikit-learn)
# This script demonstrates a complete workflow for building a sales forecasting
# model, following the user's requested steps.
# -----------------------------------------------------------------------------

# Step 1: Data Collection & Setup
# Since we don't have a file, we'll generate a synthetic time-series dataset.
# In a real-world scenario, you would use:
# df = pd.read_csv('your_sales_data.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import json

print("Generating synthetic sales data...")
# Create a date range for the data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
sales = np.random.randint(100, 250, size=len(dates))

# Add a strong yearly seasonality pattern (sinusoidal)
# Peak sales in summer months, trough in winter
seasonality = 50 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365.25)
sales = sales + seasonality

# Add a linear trend over time
trend = np.arange(len(dates)) * 0.5
sales = sales + trend

# Add some noise
noise = np.random.normal(0, 15, size=len(dates))
sales = sales + noise

df = pd.DataFrame({'Date': dates, 'Sales': sales.round(2)})

# Step 2: Exploratory Data Analysis (EDA)
print("\nPerforming Exploratory Data Analysis (EDA)...")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset information:")
print(df.info())

# Plot sales trends over time to visualize seasonality and trend
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], df['Sales'], color='skyblue', label='Sales')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# In a real project, you would also:
# - Check for missing values: df.isnull().sum()
# - Check for outliers using box plots: plt.boxplot(df['Sales'])

# Step 3: Preprocessing
print("\nPreprocessing data and engineering features...")
df['Date'] = pd.to_datetime(df['Date'])

# Extract useful features from the date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek

# Create a lagged sales feature (sales from the previous day)
# This is a powerful feature for time-series forecasting.
df['Sales_Lag1'] = df['Sales'].shift(1)
df.dropna(inplace=True) # Drop the first row which has a NaN for Sales_Lag1

# Define features and target
features = ['Year', 'Month', 'Day', 'DayOfWeek', 'Sales_Lag1']
target = 'Sales'

# Chronological Train/Test split for time series
# We split the data based on a date to prevent data leakage.
# Use 80% of data for training and 20% for testing.
split_date = df['Date'].iloc[int(len(df) * 0.8)]
train_df = df[df['Date'] < split_date]
test_df = df[df['Date'] >= split_date]

X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]

print(f"Training set size: {len(X_train)} rows")
print(f"Testing set size: {len(X_test)} rows")

# Step 4: Baseline Model
print("\nCalculating Naive Forecast Baseline...")
# A simple naive forecast: next sales = previous sales
# Since our data is daily, we use the last sales value from the training set
# as the prediction for the entire test set.
naive_predictions = np.full(shape=len(y_test), fill_value=y_train.iloc[-1])

baseline_mae = mean_absolute_error(y_test, naive_predictions)
baseline_rmse = np.sqrt(mean_squared_error(y_test, naive_predictions))
print(f"Naive Forecast Baseline MAE: {baseline_mae:.2f}")
print(f"Naive Forecast Baseline RMSE: {baseline_rmse:.2f}")

# Step 5: Regression Models (scikit-learn)
print("\nTraining Regression Models...")
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Random Forest Regressor - often provides better performance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Optional: Using TimeSeriesSplit for cross-validation
# tscv = TimeSeriesSplit(n_splits=5)
# for train_index, val_index in tscv.split(df[features]):
#     X_train_cv, X_val_cv = df[features].iloc[train_index], df[features].iloc[val_index]
#     y_train_cv, y_val_cv = df[target].iloc[train_index], df[target].iloc[val_index]
#     model.fit(X_train_cv, y_train_cv)
#     # Evaluate on X_val_cv
#     # This is a more robust way to evaluate model performance over time

# Step 6: Evaluation
print("\nEvaluating Model Performance...")

# Linear Regression Metrics
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
print(f"Linear Regression MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}")

# Random Forest Metrics
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
print(f"Random Forest MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")

# Compare to baseline
if rf_mae < baseline_mae:
    print("\nRandom Forest model outperforms the Naive baseline! ✅")
else:
    print("\nNaive baseline performs better or equal. Consider tuning the model. ⚠️")

# Step 7: Forecasting and Visualization
print("\nGenerating future forecasts and visualizing results...")

# Generate future dates for the next 90 days
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')
future_df = pd.DataFrame({'Date': future_dates})

# Populate features for the future dates
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['Day'] = future_df['Date'].dt.day
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
future_df['Sales_Lag1'] = np.nan # This will be tricky, we'll need to impute

# For a simple forecast, we'll just use the last known sales value as the lag
last_sales = df['Sales'].iloc[-1]
future_df['Sales_Lag1'].iloc[0] = last_sales

# Make predictions for the future
future_predictions = rf_model.predict(future_df[features].fillna(0)) # Using fillna for simplicity
future_df['Forecast'] = future_predictions

# Visualize the entire timeline: historical, test, and future forecast
plt.figure(figsize=(16, 9))

# Plot historical actual sales
plt.plot(df['Date'], df['Sales'], color='blue', linestyle='--', label='Historical Actual Sales')

# Plot test data actuals and predictions
plt.plot(test_df['Date'], y_test, color='green', label='Test Actual Sales', linestyle='-')
plt.plot(test_df['Date'], rf_predictions, color='red', linestyle='-', label='Test Predicted Sales')

# Plot future forecasts
plt.plot(future_df['Date'], future_df['Forecast'], color='purple', linestyle='-', label='Future Forecast')

plt.title('Sales Forecast: Historical vs. Predicted vs. Future Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# NEW: Step 8: Save data to a JSON file for the web app
print("\nSaving forecast data to JSON file...")

# Create a list of dictionaries for the historical data
historical_data = df.to_dict(orient='records')
# Create a list of dictionaries for the test data and predictions
test_data = test_df.copy()
test_data['Prediction'] = rf_predictions
test_data = test_data.to_dict(orient='records')
# Create a list of dictionaries for the future forecast
future_forecast = future_df.to_dict(orient='records')

# Prepare the data for JSON serialization
# Convert Timestamp objects to strings for JSON
for item in historical_data:
    item['Date'] = item['Date'].strftime('%Y-%m-%d')
for item in test_data:
    item['Date'] = item['Date'].strftime('%Y-%m-%d')
for item in future_forecast:
    item['Date'] = item['Date'].strftime('%Y-%m-%d')

output_data = {
    "historical": historical_data,
    "test_actual": [item['Sales'] for item in test_data],
    "test_predictions": [item['Prediction'] for item in test_data],
    "future_forecast": [item['Forecast'] for item in future_forecast],
    "test_dates": [item['Date'] for item in test_data],
    "future_dates": [item['Date'] for item in future_forecast],
    "historical_dates": [item['Date'] for item in historical_data]
}

# Save the data to a JSON file
with open('forecast_data.json', 'w') as f:
    json.dump(output_data, f, indent=4)

print("Data successfully saved to 'forecast_data.json'!")
print("\nForecasting complete. The plot shows how well the model predicts historical trends and its future sales projection.")
