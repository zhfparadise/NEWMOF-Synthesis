import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)


# 定义Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


class OptimizedFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        """
        针对Mish优化的FNN类，3层隐藏层

        修改要点:
        1. 3层隐藏层，神经元数量递减
        2. 调整dropout率
        3. 不使用额外的瓶颈层
        """
        super(OptimizedFNN, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        self.layers.append(Mish())
        self.layers.append(nn.Dropout(dropout_rate))

        # 3层隐藏层，神经元数量递减
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            self.layers.append(Mish())
            self.layers.append(nn.Dropout(dropout_rate))

        # 输出层 - 直接从最后一层隐藏层输出
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

        # 保存结构信息
        self.hidden_sizes = hidden_sizes

        # 初始化权重 - Mish对初始化更敏感
        self._initialize_weights()

    def _initialize_weights(self):
        """使用更适合Mish的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用Kaiming初始化，适合Mish
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    # 更全面的数据清洗
    # 1. 处理缺失值
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())  # 使用中位数更鲁棒

    # 2. 处理异常值（使用IQR方法）
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] > upper_bound, upper_bound,
                           np.where(df[col] < lower_bound, lower_bound, df[col]))

    # 分离特征和标签
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

    # 使用更鲁棒的标准化方法
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # 返回 x_scaler 以便后续使用
    return torch.tensor(X_scaled, dtype=torch.float32), torch.tensor(y_scaled, dtype=torch.float32), y_scaler, x_scaler


def train_model_with_early_stopping(model, train_loader, val_loader, optimizer, loss_fn, epochs, device, patience=25):
    """训练模型并实现早停机制 - 针对Mish调整了参数"""
    model.to(device)

    # 添加学习率调度器 - Mish需要更耐心的调度
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.7, verbose=True)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            # 梯度裁剪 - Mish的梯度可能更大
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # 更新学习率
        scheduler.step(val_loss)

        # 早停机制 - Mish可能需要更多epoch
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        if (epoch + 1) % 25 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 加载最佳模型
    if best_model is not None:
        model.load_state_dict(best_model)

    return model, train_losses, val_losses


def k_fold_cross_validation(X_train, y_train, hidden_sizes, k=5, batch_size=32, epochs=200, patience=25):
    """执行k折交叉验证 - 增加了epochs"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f'\n--- Fold {fold + 1}/{k} ---')

        # 创建当前折的数据集
        train_subset = Subset(TensorDataset(X_train, y_train), train_idx)
        val_subset = Subset(TensorDataset(X_train, y_train), val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # 初始化优化后的模型
        model = OptimizedFNN(input_size, hidden_sizes, output_size, dropout_rate)

        # 使用更小的学习率 - Mish需要更谨慎的学习率
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        # 训练模型
        model, train_losses, val_losses = train_model_with_early_stopping(
            model, train_loader, val_loader, optimizer, loss_fn, epochs, device, patience
        )

        # 评估当前折的验证集性能
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        # 计算指标
        mae = mean_absolute_error(val_targets, val_predictions)
        r2 = r2_score(val_targets, val_predictions)
        rmse = sqrt(mean_squared_error(val_targets, val_predictions))

        fold_results.append({
            'fold': fold + 1,
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_mae': mae,
            'val_r2': r2,
            'val_rmse': rmse
        })

        print(f'Fold {fold + 1} - Val MAE: {mae:.4f}, Val R²: {r2:.4f}, Val RMSE: {rmse:.4f}')

    return fold_results


# 参数设置 - 针对Mish优化的参数
input_size = 38
# 使用3层隐藏层，神经元从128开始递减
hidden_sizes = [128, 64, 32]  # 三层隐藏层，神经元数量递减
output_size = 1
learning_rate = 0.001  # 更小的学习率
epochs = 200  # 更多epochs
batch_size = 32
dropout_rate = 0.3  # 更高的dropout率
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
filepath = '递归式特征消除POV_重量20.csv'
X, y, y_scaler, x_scaler = load_and_preprocess_data(filepath)

# 划分训练集和测试集 (90:10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 创建训练集和测试集数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 执行5折交叉验证
print("Starting 5-fold cross validation with optimized Mish network...")
print(f"Optimized network architecture: {input_size} -> {hidden_sizes} -> {output_size}")
print("Activation function: Mish (optimized)")
print(f"Learning rate: {learning_rate}, Epochs: {epochs}, Dropout: {dropout_rate}")
fold_results = k_fold_cross_validation(X_train, y_train, hidden_sizes, k=5, batch_size=batch_size, epochs=epochs)

# 计算交叉验证的平均性能
cv_mae = np.mean([result['val_mae'] for result in fold_results])
cv_r2 = np.mean([result['val_r2'] for result in fold_results])
cv_rmse = np.mean([result['val_rmse'] for result in fold_results])

print(f'\n=== Cross Validation Results ===')
print(f'Average MAE: {cv_mae:.4f}')
print(f'Average R²: {cv_r2:.4f}')
print(f'Average RMSE: {cv_rmse:.4f}')

# 选择最佳模型（基于验证集R²最高的模型）
best_fold = max(fold_results, key=lambda x: x['val_r2'])
best_model = best_fold['model']
print(f'\nBest model from Fold {best_fold["fold"]} with R²: {best_fold["val_r2"]:.4f}')

# 使用最佳模型在训练集和测试集上评估
best_model.eval()

# 训练集预测
train_predictions = []
train_targets = []
with torch.no_grad():
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = best_model(inputs)
        train_predictions.extend(outputs.cpu().numpy())
        train_targets.extend(targets.cpu().numpy())

# 测试集预测
test_predictions = []
test_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = best_model(inputs)
        test_predictions.extend(outputs.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

# 反标准化训练集和测试集结果
train_predictions = y_scaler.inverse_transform(np.array(train_predictions).reshape(-1, 1))
train_targets = y_scaler.inverse_transform(np.array(train_targets).reshape(-1, 1))
test_predictions = y_scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
test_targets = y_scaler.inverse_transform(np.array(test_targets).reshape(-1, 1))

# 计算训练集和测试集指标
train_mae = mean_absolute_error(train_targets, train_predictions)
train_r2 = r2_score(train_targets, train_predictions)
train_rmse = sqrt(mean_squared_error(train_targets, train_predictions))

test_mae = mean_absolute_error(test_targets, test_predictions)
test_r2 = r2_score(test_targets, test_predictions)
test_rmse = sqrt(mean_squared_error(test_targets, test_predictions))

print(f'\n=== Training Set Results (using best model) ===')
print(f'Train MAE: {train_mae:.4f}')
print(f'Train R²: {train_r2:.4f}')
print(f'Train RMSE: {train_rmse:.4f}')

print(f'\n=== Test Set Results (using best model) ===')
print(f'Test MAE: {test_mae:.4f}')
print(f'Test R²: {test_r2:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')

# 绘制训练集和测试集结果在同一张图上
plt.figure(figsize=(10, 8))

# 绘制训练集结果（蓝色）
plt.scatter(train_targets, train_predictions, color='blue', alpha=0.6,
            label=f'Train (R²={train_r2:.3f}, MAE={train_mae:.3f})')

# 绘制测试集结果（红色）
plt.scatter(test_targets, test_predictions, color='red', alpha=0.6,
            label=f'Test (R²={test_r2:.3f}, MAE={test_mae:.3f})')

# 绘制理想线
all_targets = np.concatenate([train_targets, test_targets])
all_predictions = np.concatenate([train_predictions, test_predictions])
min_val = min(all_targets.min(), all_predictions.min())
max_val = max(all_targets.max(), all_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Line')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Model Performance: Training vs Test Set (3-Layer Mish: {hidden_sizes})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('train_test_comparison_3layer_mish.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制最佳fold的训练和验证损失曲线
plt.figure(figsize=(8, 5))
plt.plot(best_fold['train_losses'], label='Train Loss')
plt.plot(best_fold['val_losses'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss Curves (Best Fold {best_fold["fold"]}, 3-Layer Mish)')
plt.legend()
plt.grid(True)
plt.savefig('best_fold_loss_curves_3layer_mish.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制所有fold的验证集R²
fold_numbers = [result['fold'] for result in fold_results]
fold_r2_scores = [result['val_r2'] for result in fold_results]

plt.figure(figsize=(8, 5))
plt.bar(fold_numbers, fold_r2_scores, color='skyblue', alpha=0.7)
plt.axhline(y=cv_r2, color='red', linestyle='--', label=f'Average R²: {cv_r2:.4f}')
plt.xlabel('Fold Number')
plt.ylabel('R² Score')
plt.title(f'R² Scores for Each Fold in 5-Fold Cross Validation (3-Layer Mish)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cross_validation_r2_scores_3layerh.png', dpi=300, bbox_inches='tight')
plt.show()


# 计算网络参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


sample_model = OptimizedFNN(input_size, hidden_sizes, output_size, dropout_rate)
num_params = count_parameters(sample_model)
print(f"\n=== Network Statistics ===")
print(f"Total parameters: {num_params:,}")
print(f"Parameters per data point: {num_params / len(X_train):.2f}")
print(f"Network structure: {input_size} -> {hidden_sizes} -> {output_size}")

print(f"\n=== Optimization Summary ===")
print(f"Key changes for Mish:")
print(f"- 3-layer network: {input_size} -> {hidden_sizes} -> {output_size}")
print(f"- Lower learning rate: {learning_rate}")
print(f"- More epochs: {epochs}")
print(f"- Higher dropout: {dropout_rate}")
print(f"- Custom weight initialization")
print(f"- More patient training (patience: 25)")