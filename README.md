# 🏡 Boston Housing Price Prediction

This project demonstrates the use of various machine learning models to predict housing prices in Boston using the **Boston Housing Dataset**. The models implemented include:

1. **Simple Linear Regression (SLR)**.
2. **Multi-Layer Perceptron (MLP)** Neural Network with two hidden layers.
3. **Decision Tree Regressor**.

Each model's performance is evaluated using Mean Squared Error (MSE), and their predictions are compared visually to understand their strengths and weaknesses.

---

## 📂 Dataset

The **Boston Housing Dataset** contains 13 features describing housing and neighborhood conditions in Boston, such as:
- Crime rate.
- Number of rooms.
- Accessibility to highways.

The target variable, **MEDV**, represents the median value of owner-occupied homes in $1,000s.

---

## ⚙️ Methodology

1. **Data Preprocessing**:
   - Split into **80% training** and **20% testing** datasets.
   - Feature scaling using **StandardScaler** for improved model performance.

2. **Models Implemented**:
   - **Simple Linear Regression**: A basic model to establish a baseline.
   - **Multi-Layer Perceptron (MLP)**: A neural network with two hidden layers (64 and 32 neurons, respectively) using ReLU activation.
   - **Decision Tree Regressor**: A tree-based model with a max depth of 5 to avoid overfitting.

3. **Evaluation Metric**:
   - Mean Squared Error (MSE) on the test set.

---

## 📊 Results

| Model                 | Test Mean Squared Error |
|-----------------------|-------------------------|
| Simple Linear Regression | 24.2911               |
| Multi-Layer Perceptron  | 10.7376               |
| Decision Tree Regressor  | 8.5539                |

---

## 📈 Visualizations

### 1. Training Loss for MLP Model
The training loss decreases over epochs, indicating the model's convergence.
![MLP Training Loss](./path-to-mlp-loss.png)

---

### 2. Predicted vs True Values

#### **Simple Linear Regression**
- The predictions are less accurate, with high deviations from the true values.
![SLR Predicted vs True](./path-to-slr-scatter.png)

#### **Multi-Layer Perceptron**
- The predictions are closer to the true values, demonstrating better performance.
![MLP Predicted vs True](./path-to-mlp-scatter.png)

#### **Decision Tree Regressor**
- The most accurate predictions with minimal deviations from the true values.
![Decision Tree Predicted vs True](./path-to-dt-scatter.png)

---

### 3. Residual Plots

Residuals (differences between true and predicted values) help assess the quality of the model.

#### **Simple Linear Regression**
![SLR Residuals](./path-to-slr-residuals.png)

#### **Multi-Layer Perceptron**
![MLP Residuals](./path-to-mlp-residuals.png)

#### **Decision Tree Regressor**
![Decision Tree Residuals](./path-to-dt-residuals.png)

---

### 4. Histogram of Residuals

The distribution of residuals shows how predictions deviate from actual values.

#### **Simple Linear Regression**
![SLR Residual Histogram](./path-to-slr-histogram.png)

#### **Multi-Layer Perceptron**
![MLP Residual Histogram](./path-to-mlp-histogram.png)

#### **Decision Tree Regressor**
![Decision Tree Residual Histogram](./path-to-dt-histogram.png)

---

### 5. Line Plot of Predictions vs True Values

#### **Simple Linear Regression**
![SLR Line Plot](./path-to-slr-line.png)

#### **Multi-Layer Perceptron**
![MLP Line Plot](./path-to-mlp-line.png)

#### **Decision Tree Regressor**
![Decision Tree Line Plot](./path-to-dt-line.png)

---

## 🧪 How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/boston-housing-prediction.git
   cd boston-housing-prediction
