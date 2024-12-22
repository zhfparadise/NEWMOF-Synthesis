import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, activation_fn):
        super(FNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.Tanh())
        self.layers.append(nn.LayerNorm(hidden_size))

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation_fn())
            self.layers.append(nn.LayerNorm(hidden_size))

        self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

def load_model(model_path):
    input_size = 38  # 根据训练代码中的input_size
    hidden_size = 128
    output_size = 1
    num_hidden_layers = 3
    model = FNN(input_size, hidden_size, output_size, num_hidden_layers, Mish)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # 选择数值型特征
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 输入特征是第 2 列到倒数第二列
    numerical_features = df.iloc[:, 1:-1].values

    # 输出特征是最后一列
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # 标准化处理
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numerical_features)

    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32), y_scaler

def predict(model, X, y_scaler):
    with torch.no_grad():
        y_pred = model(X).view(-1).numpy()
    y_pred_rescaled = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    return y_pred_rescaled

# 模型和数据文件路径
model_path = 'fnn_train_GSA'
filepath = '递归式特征消除GSA20.csv'

# 加载模型
model = load_model(model_path)

# 加载和预处理数据
X, y_scaled, y_scaler = load_and_preprocess_data(filepath)

# 进行预测
y_pred_rescaled = predict(model, X, y_scaler)

# 计算并打印评价指标
mae = mean_absolute_error(y_scaled, y_pred_rescaled)
r2 = r2_score(y_scaled, y_pred_rescaled)
rmse = sqrt(mean_squared_error(y_scaled, y_pred_rescaled))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Root Mean Square Error (RMSE): {rmse}")

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(y_scaled, y_pred_rescaled, color='blue', alpha=0.5, label='Predicted vs Actual')
plt.plot([y_scaled.min(), y_scaled.max()], [y_scaled.min(), y_scaled.max()], 'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Model Prediction vs Actual')

plt.text(0.98, 0.05, f'MAE: {mae:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.98, 0.10, f'R²: {r2:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.98, 0.15, f'RMSE: {rmse:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes)

plt.legend()
plt.savefig('fnn_test_GSA.png', dpi=300)
plt.show()
