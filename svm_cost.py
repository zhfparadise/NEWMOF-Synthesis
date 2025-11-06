import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


# 读取数据集
df = pd.read_csv('price359_outlier.csv')

# 处理非数值列并转换为数值
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

for col in non_numeric_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 提取输入特征和输出特征
X = df.iloc[:, 1:-2].values
y = df.iloc[:, -1].values


# 处理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建k折交叉验证对象
kf = KFold(n_splits=5)

# 固定超参数，包括epsilon
svm_regressor = SVR(kernel='rbf', C=3000, gamma='scale', epsilon=0.1)

# 在训练集上训练模型
svm_regressor.fit(X_train, y_train)

# 预测
y_pred_train = svm_regressor.predict(X_train)
y_pred_test = svm_regressor.predict(X_test)

# 打印结果
print("训练集预测值:", y_pred_train)
print("训练集实际值:", y_train)
print("测试集预测值:", y_pred_test)
print("测试集实际值:", y_test)

# 计算模型性能
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
print(f"训练集平均绝对误差(MAE): {mae_train}")
print(f"训练集决定系数(R2): {r2_train}")
print(f"测试集平均绝对误差(MAE): {mae_test}")
print(f"测试集决定系数(R2): {r2_test}")

# 绘制预测结果图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='blue', alpha=0.5, label='Predicted vs Actual (Test)')
plt.scatter(y_train, y_pred_train, color='red', alpha=0.5, label='Predicted vs Actual (Train)')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual Cost [$/g]')
plt.ylabel('Predicted Cost [$/g]')
plt.title('SVM Model Prediction vs Actual')

plt.text(0.98, 0.05, f"Test MAE: {mae_test:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.98, 0.10, f"Test R²: {r2_test:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.98, 0.15, f"Train MAE: {mae_train:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)
plt.text(0.98, 0.20, f"Train R²: {r2_train:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)

plt.legend()
plt.savefig('price359_outlier.png', dpi=300)
plt.show()

