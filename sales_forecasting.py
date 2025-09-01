# -----------------------------------------------------------------------------
# STEP 1: Data Collection & Library Imports
# -----------------------------------------------------------------------------
# This script uses pandas for data manipulation, matplotlib/seaborn for visualization,
# and scikit-learn for machine learning models.
# Make sure these libraries are installed: pip install pandas scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# For demonstration, we'll create a synthetic dataset with a trend and seasonality.
# In a real-world scenario, you would load your data from a CSV file.
# Example: data = pd.read_csv('your_sales_data.csv')

np.random.seed(42)
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
# Create sales data with an upward trend and a seasonal component (e.g., end-of-year sales spikes)
sales = 1000 + 50 * np.arange(len(dates)) + 100 * np.sin(np.arange(len(dates)) * np.pi / 6) + np.random.normal(0, 50, len(dates))
sales_data = pd.DataFrame({'Date': dates, 'Sales': sales.round(2)})

print("Initial Data Snapshot:")
print(sales_data.head())
print("\n")

# -----------------------------------------------------------------------------
# STEP 2: Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------
print("Performing EDA...")

# Plotting sales trends over time to identify trends and seasonality.
plt.figure(figsize=(14, 7))
sns.lineplot(x='Date', y='Sales', data=sales_data, label='Sales Trend')
plt.title('Sales Trends Over Time', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.grid(True)
plt.show()

# Checking for missing values
print("Missing values in the dataset:")
print(sales_data.isnull().sum())
print("\n")

# Identifying outliers using a box plot
plt.figure(figsize=(10, 5))
sns.boxplot(x=sales_data['Sales'])
plt.title('Box Plot of Sales', fontsize=16)
plt.xlabel('Sales', fontsize=12)
plt.show()

# -----------------------------------------------------------------------------
# STEP 3: Preprocessing and Feature Engineering
# -----------------------------------------------------------------------------
print("Preprocessing data and engineering features...")

# Convert 'Date' to datetime and extract time-based features
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
sales_data['Year'] = sales_data['Date'].dt.year
sales_data['Month'] = sales_data['Date'].dt.month
sales_data['DayOfWeek'] = sales_data['Date'].dt.dayofweek

# Create lagged sales features, which are crucial for time series forecasting.
# These features represent the sales from previous periods.
sales_data['Lag_1'] = sales_data['Sales'].shift(1)
sales_data['Lag_3'] = sales_data['Sales'].shift(3)
sales_data['Lag_6'] = sales_data['Sales'].shift(6)

# Drop initial rows with NaN values created by the shift operation
sales_data.dropna(inplace=True)
sales_data.reset_index(drop=True, inplace=True)

# Define features (X) and target (y)
features = ['Year', 'Month', 'DayOfWeek', 'Lag_1', 'Lag_3', 'Lag_6']
target = 'Sales'
X = sales_data[features]
y = sales_data[target]

# Perform a chronological train/test split. It's important to not shuffle time series data.
split_point = int(len(sales_data) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("\n")

# -----------------------------------------------------------------------------
# STEP 4: Baseline Model (Naïve Forecast)
# -----------------------------------------------------------------------------
print("Running baseline model (Naïve Forecast)...")

# The Naïve forecast assumes the next value will be the last observed value.
# For our features, we can use the 'Lag_1' feature for this.
y_pred_naive = X_test['Lag_1']

mae_naive = mean_absolute_error(y_test, y_pred_naive)
rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))

print(f"Naïve Forecast Metrics:")
print(f"  MAE (Mean Absolute Error): {mae_naive:.2f}")
print(f"  RMSE (Root Mean Squared Error): {rmse_naive:.2f}")
print("\n")

# -----------------------------------------------------------------------------
# STEP 5: Regression Models (scikit-learn)
# -----------------------------------------------------------------------------
print("Training Linear Regression and Random Forest models...")

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Use TimeSeriesSplit for more robust cross-validation on Random Forest
tscv = TimeSeriesSplit(n_splits=5)
cv_scores = []
for train_index, test_index in tscv.split(X):
    X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
    y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]
    
    rf_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = rf_model.predict(X_test_cv)
    
    cv_scores.append(mean_absolute_error(y_test_cv, y_pred_cv))

print("Random Forest Regressor Cross-Validation MAE scores:")
print(np.round(cv_scores, 2))
print(f"Average CV MAE: {np.mean(cv_scores):.2f}")
print("\n")

# -----------------------------------------------------------------------------
# STEP 6: Evaluation and Comparison
# -----------------------------------------------------------------------------
print("Evaluating model performance...")

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Presenting the results in a clear, comparative format
print("----------------------------------------------------------")
print(f"| Model               | MAE (Lower is better) | RMSE (Lower is better) |")
print("----------------------------------------------------------")
print(f"| Naïve Forecast      | {mae_naive:<21.2f} | {rmse_naive:<22.2f} |")
print(f"| Linear Regression   | {mae_lr:<21.2f} | {rmse_lr:<22.2f} |")
print(f"| Random Forest       | {mae_rf:<21.2f} | {rmse_rf:<22.2f} |")
print("----------------------------------------------------------")
print("\n")

# -----------------------------------------------------------------------------
# STEP 7: Forecasting and Visualization
# -----------------------------------------------------------------------------
print("Generating future predictions and visualizing results...")

# Create a DataFrame for the next 6 months of data
last_date = sales_data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=7, freq='M')[1:]
future_df = pd.DataFrame({'Date': future_dates})

# Engineer features for the future data using the last known values
future_df['Year'] = future_df['Date'].dt.year
future_df['Month'] = future_df['Date'].dt.month
future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek

# Use the last few months of actual sales data to create the lagged features for forecasting
future_df['Lag_1'] = sales_data['Sales'].iloc[-1]
future_df['Lag_3'] = sales_data['Sales'].iloc[-3]
future_df['Lag_6'] = sales_data['Sales'].iloc[-6]

# Use the best-performing model (Random Forest) to predict future sales
X_future = future_df[features]
future_predictions = rf_model.predict(X_future)
future_df['Predicted_Sales'] = future_predictions

# Visualize the actual sales data vs. the new predictions
plt.figure(figsize=(14, 8))
plt.plot(sales_data['Date'], sales_data['Sales'], label='Actual Sales', color='blue')
plt.plot(future_df['Date'], future_df['Predicted_Sales'], label='Predicted Sales', color='red', linestyle='--')
plt.title('Actual Sales vs. Future Forecast', fontsize=18)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

print("Future Sales Predictions (Next 6 Months):")
print(future_df[['Date', 'Predicted_Sales']].round(2))
