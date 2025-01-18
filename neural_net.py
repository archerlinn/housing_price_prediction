import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ================
# 1. LOAD & PREP
# ================
boston = pd.read_csv("/Users/archer/code/housing_price_prediction/boston.csv")

target_column = 'MEDV'
data = boston.drop(columns=[target_column]).values  # Features
target = boston[target_column].values              # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ================
# 2. DEFINE MODEL
# ================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.network = nn.Sequential(
            #nn.Linear(input_dim, 128),  # Hidden layer 1 (add to test)
            #nn.ReLU(),
            nn.Linear(input_dim, 64),        # Hidden layer 2 (change input_dim to 128 if add another layer)
            nn.ReLU(),
            nn.Linear(64, 32),         # Hidden layer 3
            nn.ReLU(),
            nn.Linear(32, 1)           # Output layer
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
input_dim = X_train.shape[1]
model = MLPRegressor(input_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # smaller LR

# ================
# 3. TRAIN MODEL
# ================
epochs = 1000
losses = []

for epoch in range(epochs):
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot training loss
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss")
plt.show()

# ================
# 4. EVALUATE MODEL
# ================
model.eval()
with torch.no_grad():
    # Get predictions as NumPy array
    test_predictions = model(X_test_tensor).flatten().detach().numpy()
    
    # Convert test_predictions back to a PyTorch tensor
    test_predictions_tensor = torch.tensor(test_predictions, dtype=torch.float32)
    
    test_loss = criterion(test_predictions_tensor, y_test_tensor.flatten())
    print(f"Test Mean Squared Error: {test_loss.item():.4f}")

residuals = y_test - test_predictions

# Plot predictions vs true values
# 1. Predicted vs True Values Scatter Plot with Error Bars
plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.6, label="Predictions")
plt.plot([0, 50], [0, 50], color='red', linestyle="--", label="Perfect Prediction")
plt.errorbar(y_test, test_predictions, yerr=abs(residuals), fmt='o', alpha=0.3, label="Error Bars")
plt.xlabel("True Values [Price]")
plt.ylabel("Predicted Values [Price]")
plt.title("True vs Predicted Values with Error Bars")
plt.legend()
plt.grid(True)
plt.show()

# 2. Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(test_predictions, residuals, alpha=0.6, label="Residuals")
plt.axhline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Predicted Values [Price]")
plt.ylabel("Residuals [Price]")
plt.title("Residual Plot")
plt.legend()
plt.grid(True)
plt.show()

# 3. Histogram of Residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color="blue", alpha=0.7)
plt.axvline(0, color='red', linestyle="--", label="Zero Error Line")
plt.xlabel("Residuals [Price]")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.legend()
plt.grid(True)
plt.show()

# 4. Line Plot of Predictions vs True Values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label="True Values", linestyle="--", marker="o")
plt.plot(range(len(test_predictions)), test_predictions, label="Predicted Values", linestyle="--", marker="o")
plt.xlabel("Test Data Index")
plt.ylabel("Price")
plt.title("True vs Predicted Values Over Test Set")
plt.legend()
plt.grid(True)
plt.show()
