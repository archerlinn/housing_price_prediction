# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Initialize the model, define the loss function and the optimizer
input_dim = X_train.shape[1]
model = LinearRegressionModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Train the model
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot the training loss
plt.figure(figsize=(8, 6))
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor).flatten().detach().numpy()
    test_loss = criterion(torch.tensor(test_predictions), y_test_tensor.flatten()).item()
    print(f"Test Mean Squared Error: {test_loss:.4f}")

# Calculate residuals
residuals = y_test - test_predictions

# Visualizations
# 1. Predicted vs True Values Scatter Plot with Error Bars
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.6, label="Predictions")
plt.plot([0, 50], [0, 50], color='red', linestyle="--", label="Perfect Prediction")
plt.errorbar(y_test, test_predictions, yerr=abs(residuals), fmt='o', alpha=0.3, label="Error Bars")
plt.xlabel("True Values [Price]")
plt.ylabel("Predicted Values [Price]")
plt.title("True vs Predicted Values with Error Bars (Simple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()

# 2. Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions, residuals, alpha=0.6, label="Residuals")
plt.axhline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Predicted Values [Price]")
plt.ylabel("Residuals [Price]")
plt.title("Residual Plot (Simple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color="blue", alpha=0.7)
plt.axvline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Residuals [Price]")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Simple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()

# 4. Line Plot of Predictions vs True Values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="True Values", linestyle="--", marker="o")
plt.plot(range(len(test_predictions)), test_predictions, label="Predicted Values", linestyle="--", marker="o")
plt.xlabel("Test Data Index")
plt.ylabel("Price")
plt.title("True vs Predicted Values Over Test Set (Simple Linear Regression)")
plt.legend()
plt.grid(True)
plt.show()
