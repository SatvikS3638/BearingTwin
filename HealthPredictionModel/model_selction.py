import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb  # Import XGBoost

# --- Load your data ---
df = pd.read_csv("all_features.csv")

# --- Select features and label ---
X = df.drop(["severity", "segment", "day"], axis=1)  # Drop non-feature columns
y = df["severity"]

# --- Scale the features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # scale for better performance and convergence

# --- Define models ---
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": xgb.XGBRegressor(random_state=42),  # Add XGBoost
    "SVR": SVR(),
    "MLP (Simple)": MLPRegressor(random_state=42, max_iter=500, early_stopping=True),  # Simple MLP
    "MLP (Complex)": MLPRegressor(random_state=42, max_iter=500, early_stopping=True,
                                  hidden_layer_sizes=(128, 64, 32)),  # More complex MLP
    "MLP (More Neurons)": MLPRegressor(random_state=42, max_iter=500, early_stopping=True,
                                       hidden_layer_sizes=(256, 128)),  # Another complex MLP
}

# --- Cross-validation setup ---
n_splits = 5  # Number of folds for cross-validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# --- Store results ---
results = {}
for name in models:
    results[name] = {
        "r2": [],
        "rmse": []
    }

# --- Perform cross-validation ---
for name, model in models.items():
    print(f"Training and evaluating {name}...")
    # Use scaled data X_scaled
    r2_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
    rmse_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')  # Note the 'neg_'
    rmse_scores = np.sqrt(-rmse_scores)  # convert back to positive

    results[name]["r2"] = r2_scores
    results[name]["rmse"] = rmse_scores

    print(f"{name} - R^2 Mean: {r2_scores.mean():.4f}, RMSE Mean: {rmse_scores.mean():.4f}")

# --- Visualize results ---

# --- Box Plots ---
# R^2 scores (Box Plot)
plt.figure(figsize=(12, 7))
plt.boxplot([results[name]["r2"] for name in models], labels=models.keys())
plt.title("Cross-Validation R^2 Scores (Box Plot)")
plt.ylabel("R^2 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# RMSE scores (Box Plot)
plt.figure(figsize=(12, 7))
plt.boxplot([results[name]["rmse"] for name in models], labels=models.keys())
plt.title("Cross-Validation RMSE Scores (Box Plot)")
plt.ylabel("RMSE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


