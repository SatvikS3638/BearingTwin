import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib  # For saving the model

# --- Load your data ---
df = pd.read_csv(r"all_features.csv")

# --- Select features and label ---
X = df.drop(["severity", "segment", "day"], axis=1)  # Drop non-feature columns
y = df["severity"]


# --- Train the Random Forest model on the entire dataset ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters
rf_model.fit(X, y)

# --- Save the trained model to a file ---
model_filename = "random_forest_model.joblib"
joblib.dump(rf_model, model_filename)

print(f"✅ Trained Random Forest model saved to '{model_filename}'")