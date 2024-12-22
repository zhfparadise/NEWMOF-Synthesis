import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt

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


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)  # 读取CSV文件
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    numerical_features = df.iloc[:, 1:-2].values  # 输入特征是第二列到倒数第二列
    X = pd.DataFrame(numerical_features).values
    y = df.iloc[:, -2].values.reshape(-1, 1)  # 输出特征是最后一列
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32), y_scaler


def train_and_evaluate_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    val_maes = []
    val_r2s = []
    val_rmses = []
    train_predictions = []
    train_targets = []
    test_predictions = []
    test_targets = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_predictions.extend(outputs.view(-1).tolist())
            train_targets.extend(targets.view(-1).tolist())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                val_predictions.extend(outputs.view(-1).tolist())
                val_targets.extend(targets.view(-1).tolist())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        mae = mean_absolute_error(val_targets, val_predictions)
        r2 = r2_score(val_targets, val_predictions)
        rmse = sqrt(mean_squared_error(val_targets, val_predictions))
        val_maes.append(mae)
        val_r2s.append(r2)
        val_rmses.append(rmse)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val MAE: {mae}, Val R²: {r2}, Val RMSE: {rmse}')

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_predictions.extend(outputs.view(-1).tolist())
            test_targets.extend(targets.view(-1).tolist())

    return train_losses, val_losses, val_maes, val_r2s, val_rmses, train_predictions, train_targets, test_predictions, test_targets


# 设置模型参数
input_size = 36  # 输入特征维度根据数据集特征数量调整
hidden_size = 128
output_size = 1
num_hidden_layers = 3
learning_rate = 0.001
epochs = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FNN(input_size, hidden_size, output_size, num_hidden_layers, Mish)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

filepath = 'price359_outlier.csv'  # 替换成你的数据集路径
X, y, y_scaler = load_and_preprocess_data(filepath)

dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_losses, val_losses, val_maes, val_r2s, val_rmses, train_predictions, train_targets, test_predictions, test_targets = train_and_evaluate_model(model, train_loader, val_loader, test_loader, optimizer, loss_fn, epochs, device)

# 训练完成后，保存模型
torch.save(model.state_dict(), 'fnn_train_GSA')

# 反标准化训练和测试结果
train_predictions = y_scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1))
train_targets = y_scaler.inverse_transform(np.array(train_targets).reshape(-1, 1))
test_predictions = y_scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
test_targets = y_scaler.inverse_transform(np.array(test_targets).reshape(-1, 1))

# 计算测试集的评价指标
mae = mean_absolute_error(test_targets, test_predictions)
r2 = r2_score(test_targets, test_predictions)
rmse = sqrt(mean_squared_error(test_targets, test_predictions))

print(f"Test Mean Absolute Error (MAE): {mae}")
print(f"Test R-squared (R²): {r2}")
print(f"Test Root Mean Square Error (RMSE): {rmse}")

# 计算训练集的评价指标
train_mae = mean_absolute_error(train_targets, train_predictions)
train_r2 = r2_score(train_targets, train_predictions)
train_rmse = sqrt(mean_squared_error(train_targets, train_predictions))

# 绘制训练和测试结果图
plt.figure(figsize=(8, 6))
plt.scatter(train_targets, train_predictions, color='green', alpha=0.5, label='Train: Predicted vs Actual')
plt.scatter(test_targets, test_predictions, color='blue', alpha=0.5, label='Test: Predicted vs Actual')
plt.plot([min(train_targets.min(), test_targets.min()), max(train_targets.max(), test_targets.max())],
         [min(train_targets.min(), test_targets.min()), max(train_targets.max(), test_targets.max())],
         'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Model Prediction vs Actual')

# 显示测试集的评价指标
plt.text(0.98, 0.30, f'Test R²: {r2:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='blue')
plt.text(0.98, 0.25, f'Test MAE: {mae:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='blue')
plt.text(0.98, 0.20, f'Test RMSE: {rmse:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='blue')

# 显示训练集的评价指标
plt.text(0.98, 0.15, f'Train R²: {train_r2:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='green')
plt.text(0.98, 0.10, f'Train MAE: {train_mae:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='green')
plt.text(0.98, 0.05, f'Train RMSE: {train_rmse:.2f}', ha='right', va='bottom', transform=plt.gca().transAxes, color='green')

plt.legend()
plt.savefig('fnn_test_price359_outlier.png', dpi=300)
plt.show()
