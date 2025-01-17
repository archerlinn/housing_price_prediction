# Import required libraries
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
boston = pd.read_csv("/Users/archer/code/slr_houseprice_prediction/boston.csv")

target_column = 'MEDV'
data = boston.drop(columns=[target_column]).values  # Features
target = boston[target_column].values  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Train a Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)  # Adjust max_depth as needed
dt_model.fit(X_train, y_train)

# Make predictions
train_predictions = dt_model.predict(X_train)
test_predictions = dt_model.predict(X_test)

# Evaluate performance
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print(f"Train Mean Squared Error: {train_mse:.4f}")
print(f"Test Mean Squared Error: {test_mse:.4f}")

# Residuals
residuals = y_test - test_predictions

# Visualizations
# 1. Predicted vs True Values Scatter Plot with Error Bars
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.6, label="Predictions")
plt.plot([0, 50], [0, 50], color='red', linestyle="--", label="Perfect Prediction")
plt.errorbar(y_test, test_predictions, yerr=abs(residuals), fmt='o', alpha=0.3, label="Error Bars")
plt.xlabel("True Values [Price]")
plt.ylabel("Predicted Values [Price]")
plt.title("True vs Predicted Values with Error Bars (Decision Tree)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions, residuals, alpha=0.6, label="Residuals")
plt.axhline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Predicted Values [Price]")
plt.ylabel("Residuals [Price]")
plt.title("Residual Plot (Decision Tree)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color="blue", alpha=0.7)
plt.axvline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Residuals [Price]")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Decision Tree)")
plt.legend()
plt.grid(True)
plt.show()

# 4. Line Plot of Predictions vs True Values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="True Values", linestyle="--", marker="o")
plt.plot(range(len(test_predictions)), test_predictions, label="Predicted Values", linestyle="--", marker="x")
plt.xlabel("Test Data Index")
plt.ylabel("Price")
plt.title("True vs Predicted Values Over Test Set (Decision Tree)")
plt.legend()
plt.grid(True)
plt.show()
