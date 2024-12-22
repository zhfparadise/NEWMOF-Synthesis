import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
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

    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32)


def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, device):
    model.to(device)
    train_losses = []
    val_maes = []
    val_r2s = []
    val_rmses = []

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

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.extend(outputs.view(-1).tolist())
                val_targets.extend(targets.view(-1).tolist())

        mae = mean_absolute_error(val_targets, val_predictions)
        r2 = r2_score(val_targets, val_predictions)
        rmse = sqrt(mean_squared_error(val_targets, val_predictions))
        val_maes.append(mae)
        val_r2s.append(r2)
        val_rmses.append(rmse)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss}, Val MAE: {mae}, Val R²: {r2}, Val RMSE: {rmse}')

    return train_losses, val_maes, val_r2s, val_rmses


# 设置模型参数
input_size = 38
hidden_size = 128
output_size = 1
num_hidden_layers = 3
learning_rate = 0.001
epochs = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FNN(input_size, hidden_size, output_size, num_hidden_layers, Mish)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

filepath = '递归式特征消除GSA20.csv'
X, y = load_and_preprocess_data(filepath)

dataset = TensorDataset(X, y)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

train_losses, val_maes, val_r2s, val_rmses = train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs,
                                                         device)

# 训练完成后，保存模型
torch.save(model.state_dict(), 'fnn_train_GSA')

# 绘制训练和验证结果图
plt.figure(figsize=(12, 8))

# 绘制训练损失
plt.subplot(2, 1, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# 绘制验证指标
plt.subplot(2, 1, 2)
plt.plot(val_maes, label='Val MAE')
plt.plot(val_r2s, label='Val R²')
plt.plot(val_rmses, label='Val RMSE')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Validation Metrics')
plt.legend()

plt.tight_layout()
plt.savefig('GSA20.png', dpi=300)
plt.show()
