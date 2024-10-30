from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import RandomizedSearchCV  # Thêm dòng này
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

app = FastAPI()

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("./Data.csv")
df.columns = ["Index", "Height", "Weight"]

# Tạo dữ liệu x và y
x = df["Height"].values
y = df["Weight"].values

# Tổng số mẫu
N = x.shape[0]

# Tính toán hệ số cho Hồi quy tuyến tính
m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
b = (np.sum(y) - m * np.sum(x)) / N

# Hàm dự đoán của hồi quy tuyến tính
def predict_weight(height: float) -> float:
    return m * height + b

# Tìm alpha tốt nhất cho Ridge Regression
def find_best_alpha():
    ridge_model = Ridge()
    param_dist = {'alpha': np.logspace(-3, 3, 100)}
    search = RandomizedSearchCV(ridge_model, param_distributions=param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', random_state=42)
    search.fit(x.reshape(-1, 1), y)
    return search.best_params_['alpha']

# Tìm alpha tốt nhất trước khi huấn luyện mô hình
best_alpha = find_best_alpha()

# Hàm dự đoán cho Ridge Regression
def predict_weight_ridge(height: float) -> float:
    ridge_m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) + best_alpha * N - (np.sum(x) ** 2))
    ridge_b = (np.sum(y) - ridge_m * np.sum(x)) / N
    return ridge_m * height + ridge_b

# Huấn luyện Neural Network
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)
neural_model.fit(x.reshape(-1, 1), y)

# Tạo mô hình stacking
class StackingModel:
    def __init__(self):
        self.linear_model = LinearRegression()
        self.ridge_model = Ridge(alpha=best_alpha)
        self.neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=0)

        # Huấn luyện các mô hình
        self.linear_model.fit(x.reshape(-1, 1), y)
        self.ridge_model.fit(x.reshape(-1, 1), y)
        self.neural_model.fit(x.reshape(-1, 1), y)

    def predict(self, height):
        linear_pred = self.linear_model.predict(np.array([[height]]))[0]
        ridge_pred = self.ridge_model.predict(np.array([[height]]))[0]
        neural_pred = self.neural_model.predict(np.array([[height]]))[0]

        return np.mean([linear_pred, ridge_pred, neural_pred])



stacking_model = StackingModel()

# Định nghĩa lớp dữ liệu đầu vào
class PredictionInput(BaseModel):
    height: float

# Hàm đánh giá các mô hình
def evaluate_models():
    y_pred_linear = [predict_weight(h) for h in x]
    y_pred_ridge = [predict_weight_ridge(h) for h in x]
    y_pred_neural = neural_model.predict(x.reshape(-1, 1))
    y_pred_stacking = [stacking_model.predict(h) for h in x]

    metrics = {
        "Linear": {
            "MSE": mean_squared_error(y, y_pred_linear),
            "MAE": mean_absolute_error(y, y_pred_linear),
        },
        "Ridge": {
            "MSE": mean_squared_error(y, y_pred_ridge),
            "MAE": mean_absolute_error(y, y_pred_ridge),
        },
        "Neural": {
            "MSE": mean_squared_error(y, y_pred_neural),
            "MAE": mean_absolute_error(y, y_pred_neural),
        },
        "Stacking": {
            "MSE": mean_squared_error(y, y_pred_stacking),
            "MAE": mean_absolute_error(y, y_pred_stacking),
        }
    }
    return metrics

@app.post("/predict")
async def predict(input_data: PredictionInput):
    height = input_data.height

    predicted_weight_linear = predict_weight(height)
    predicted_weight_ridge = predict_weight_ridge(height)
    predicted_weight_neural = neural_model.predict(np.array([[height]]))[0]
    predicted_weight_stacking = stacking_model.predict(height)

    metrics = evaluate_models()

    return {
        "predicted_weight_linear": predicted_weight_linear,
        "predicted_weight_ridge": predicted_weight_ridge,
        "predicted_weight_neural": predicted_weight_neural,
        "predicted_weight_stacking": predicted_weight_stacking,
        "metrics": metrics,
    }

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <html>
        <head>
            <title>Dự đoán cân nặng</title>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        </head>
        <body>
            <div class="container">
                <div class="row justify-content-center align-items-center" style="height: 100vh;">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-body">
                                <h2 class="card-title text-center">Dự đoán cân nặng</h2>
                                <form id="predictionForm">
                                    <div class="form-group">
                                        <label for="height">Chiều cao (cm):</label>
                                        <input type="number" id="height" name="height" class="form-control" step="any" required>
                                    </div>
                                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                                </form>
                                <div id="result" class="mt-3"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                async function predict() {
                    const height = document.getElementById('height').value;

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ height: parseFloat(height) }),
                    });

                    const data = await response.json();
                    document.getElementById('result').innerHTML = `
                        <p>Cân nặng dự đoán theo Hồi quy tuyến tính: ${data.predicted_weight_linear.toFixed(2)} kg</p>
                        <p>Cân nặng dự đoán theo Hồi quy Ridge: ${data.predicted_weight_ridge.toFixed(2)} kg</p>
                        <p>Cân nặng dự đoán theo Neural Network: ${data.predicted_weight_neural.toFixed(2)} kg</p>
                        <p>Cân nặng dự đoán theo Stacking: ${data.predicted_weight_stacking.toFixed(2)} kg</p>
                        <p><strong>Đánh giá mô hình:</strong></p>
                        <p>Hồi quy tuyến tính - MSE: ${data.metrics.Linear.MSE.toFixed(2)}, MAE: ${data.metrics.Linear.MAE.toFixed(2)}</p>
                        <p>Hồi quy Ridge - MSE: ${data.metrics.Ridge.MSE.toFixed(2)}, MAE: ${data.metrics.Ridge.MAE.toFixed(2)}</p>
                        <p>Neural Network - MSE: ${data.metrics.Neural.MSE.toFixed(2)}, MAE: ${data.metrics.Neural.MAE.toFixed(2)}</p>
                        <p>Stacking - MSE: ${data.metrics.Stacking.MSE.toFixed(2)}, MAE: ${data.metrics.Stacking.MAE.toFixed(2)}</p>
                    `;
                }
            </script>
        </body>
    </html>
    """
