import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# --- Load the data ---
df = pd.read_csv(r"all_features.csv")

# --- Prepare data ---
# Features (excluding target features, severity, segment, and day)
X = df.drop(["severity", "segment", "day", "health_indicator", "kurtosis", "envelope_spectrum_peak"], axis=1)
# Target variables
# These are now the *smoothed* versions (as they are used in the HI calculation)
y_kurtosis = df["kurtosis"]  # Keep the raw values initially
y_envelope = df["envelope_spectrum_peak"]  # Keep the raw values initially

# --- Group by day and compute mean values *before* splitting, to create the smoothed targets ---
grouped = df.groupby('day').agg({
    'kurtosis': 'mean',
    'envelope_spectrum_peak': 'mean'
}).reset_index()

# Smooth both signals using rolling mean
grouped['kurtosis_smoothed'] = grouped['kurtosis'].rolling(window=3, min_periods=1).mean()
grouped['envelope_smoothed'] = grouped['envelope_spectrum_peak'].rolling(window=3, min_periods=1).mean()

# Merge the smoothed values back into the main DataFrame, aligning on 'day'
df = pd.merge(df, grouped[['day', 'kurtosis_smoothed', 'envelope_smoothed']], on='day', how='left')


# Now set the targets to the *smoothed* values
y_kurtosis = df["kurtosis_smoothed"]
y_envelope = df["envelope_smoothed"]
# Health indicator will be calculated *after* prediction

# --- Split data into training and testing sets ---
X_train, X_test, y_kurtosis_train, y_kurtosis_test, y_envelope_train, y_envelope_test = train_test_split(
    X, y_kurtosis, y_envelope, test_size=0.2, random_state=42
)

# --- Train Random Forest models for kurtosis and envelope spectrum peak ---
# Kurtosis model
rf_model_kurtosis = RandomForestRegressor(random_state=42)
rf_model_kurtosis.fit(X_train, y_kurtosis_train)

# Envelope spectrum peak model
rf_model_envelope = RandomForestRegressor(random_state=42)
rf_model_envelope.fit(X_train, y_envelope_train)


# --- Make predictions ---
y_kurtosis_pred = rf_model_kurtosis.predict(X_test)
y_envelope_pred = rf_model_envelope.predict(X_test)

# --- Calculate Health Indicator using predicted values ---
# Define weights (same as original HI calculation)
w1 = 0.5
w2 = 0.5

# Normalize the predicted values (very important!)
kurtosis_norm_pred = (y_kurtosis_pred - y_kurtosis_pred.min()) / (y_kurtosis_pred.max() - y_kurtosis_pred.min())
envelope_norm_pred = (y_envelope_pred - y_envelope_pred.min()) / (y_envelope_pred.max() - y_envelope_pred.min())

# Calculate the predicted health indicator
y_health_pred = w1 * kurtosis_norm_pred + w2 * envelope_norm_pred


# --- Evaluate models ---
# Kurtosis
mse_kurtosis = mean_squared_error(y_kurtosis_test, y_kurtosis_pred)
r2_kurtosis = r2_score(y_kurtosis_test, y_kurtosis_pred)
print("Kurtosis Model:")
print(f"  Mean Squared Error: {mse_kurtosis:.4f}")
print(f"  R^2 Score: {r2_kurtosis:.4f}")

# Envelope spectrum peak
mse_envelope = mean_squared_error(y_envelope_test, y_envelope_pred)
r2_envelope = r2_score(y_envelope_test, y_envelope_pred)
print("Envelope Spectrum Peak Model:")
print(f"  Mean Squared Error: {mse_envelope:.4f}")
print(f"  R^2 Score: {r2_envelope:.4f}")


# --- Calculate Health Indicator using *actual* values ---
# First, normalize the actual values
kurtosis_norm_actual = (y_kurtosis_test - y_kurtosis_test.min()) / (y_kurtosis_test.max() - y_kurtosis_test.min())
envelope_norm_actual = (y_envelope_test - y_envelope_test.min()) / (y_envelope_test.max() - y_envelope_test.min())

# Calculate the *actual* health indicator using the actual smoothed values
y_health_actual = w1 * kurtosis_norm_actual + w2 * envelope_norm_actual

# --- Evaluate the Health Indicator (using *both* predicted and actual) ---
mse_health = mean_squared_error(y_health_actual, y_health_pred)
r2_health = r2_score(y_health_actual, y_health_pred) #compare it to actual
print("Health Indicator Model (Calculated from Predictions):")
print(f"  Mean Squared Error: {mse_health:.4f}")
print(f"  R^2 Score: {r2_health:.4f}")


# --- Plot actual vs predicted values as line graph ---
num_points = 50 #plot only the first 50.

# Kurtosis
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.plot(range(num_points), y_kurtosis_test[:num_points], marker='x', linestyle='-', label='Actual Kurtosis', color='blue')
plt.plot(range(num_points), y_kurtosis_pred[:num_points], marker='o', linestyle='-', label='Predicted Kurtosis', color='red', alpha=0.7) # Add transparency
plt.xlabel("Test Sample Index")
plt.ylabel("Kurtosis")
plt.title("Actual vs Predicted Kurtosis (Line Graph)")
plt.grid(True)
plt.legend()
plt.tight_layout()  # Improve spacing
plt.show()


# Envelope spectrum peak
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.plot(range(num_points), y_envelope_test[:num_points], marker='x', linestyle='-', label='Actual Envelope Peak', color='blue')
plt.plot(range(num_points), y_envelope_pred[:num_points], marker='o', linestyle='-', label='Predicted Envelope Peak', color='red', alpha=0.7) # Add transparency
plt.xlabel("Test Sample Index")
plt.ylabel("Envelope Spectrum Peak")
plt.title("Actual vs Predicted Envelope Spectrum Peak (Line Graph)")
plt.grid(True)
plt.legend()
plt.tight_layout()  # Improve spacing
plt.show()

# Health Indicator
plt.figure(figsize=(12, 6))  # Adjust figure size for better readability
plt.plot(range(num_points), y_health_actual[:num_points], marker='x', linestyle='-', label='Actual Health Indicator', color='blue')
plt.plot(range(num_points), y_health_pred[:num_points], marker='o', linestyle='-', label='Predicted Health Indicator', color='red', alpha=0.7) # Add transparency
plt.xlabel("Test Sample Index")
plt.ylabel("Health Indicator")
plt.title("Actual vs Predicted Health Indicator (Line Graph)")
plt.grid(True)
plt.legend()
plt.tight_layout()  # Improve spacing
plt.show()

# --- Plot Actual vs Predicted Health Indicator against Days ---
# Create a DataFrame for the test set predictions to easily group by day
test_df = pd.DataFrame({'day': df["day"].iloc[X_test.index],
                        'actual_health': y_health_actual,
                        'predicted_health': y_health_pred})

# Group by day and calculate the mean actual and predicted health values
grouped_health = test_df.groupby('day')[['actual_health', 'predicted_health']].mean()

# Smooth the predicted health indicator values
grouped_health['predicted_health_smoothed'] = grouped_health['predicted_health'].rolling(window=3, min_periods=1).mean()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(grouped_health.index, grouped_health['actual_health'], marker='x', linestyle='-', label='Actual Health Indicator', color='blue')
plt.plot(grouped_health.index, grouped_health['predicted_health_smoothed'], marker='o', linestyle='-', label='Predicted Health Indicator', color='green', alpha=0.7)  # Change color
plt.xlabel("Day")
plt.ylabel("Health Indicator")
plt.title("Actual vs Predicted Health Indicator Over Days")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()