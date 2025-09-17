import pandas as pd
import json
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from dvclive import Live

# Function to calculate RMSLE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Load the pipeline
pipeline = joblib.load("models/pipeline.joblib")

# Read the test data
test = pd.read_csv("data/test.csv").dropna()

# Assuming 'text_column' is the column with text data and 'Times' is the target
X_test = test["Password"]
y_test = test["Times"]

# Make predictions
predictions = pipeline.predict(X_test)
print(f"Predictions: {predictions}")

# Calculate RMSE
rmse_score = rmse(y_test, predictions)
print(f"RMSE: {rmse_score}")

metrics = {"RMSE": rmse_score}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

with Live() as live:
    live.log_metric("RMSE", rmse_score)
