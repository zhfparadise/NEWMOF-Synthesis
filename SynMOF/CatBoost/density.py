import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 加载数据
data = pd.read_csv('递归式特征消除density20.csv')

# 分割数据为输入特征和输出特征
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 将数据分为训练集和测试集，90:10比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义类别特征的索引
cat_features = []

# 创建CatBoost Pool（包含类别特征的索引）
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# 使用提供的最佳参数初始化CatBoost回归器
model = CatBoostRegressor(
    iterations=150,
    learning_rate=0.1,  # 可调整，通常0.1是一个好的起点
    depth=7,
    l2_leaf_reg=3,
    loss_function='RMSE',
    verbose=100
)

# 训练模型
model.fit(train_pool)

# 进行预测
y_pred_train = model.predict(train_pool)
y_pred_test = model.predict(test_pool)

# 评估模型
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'Mean Absolute Error on train set: {mae_train}, R² on train set: {r2_train}')
print(f'Mean Absolute Error on test set: {mae_test}, R² on test set: {r2_test}')

# 确定绘制参考线的范围
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())

# 设置字体大小
plt.rcParams.update({'font.size': 18})

# 绘制实际值与预测值的散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_pred_train, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, alpha=0.5, color='red', label='Test')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # 画y=x的参考线
plt.xlabel('Actual Density [g/cm³]')
plt.ylabel('Predicted Density [g/cm³]')


# 添加评估指标文本
plt.text(0.02, 0.97, f'Train MAE: {mae_train:.2f}, Train R²: {r2_train:.2f}', ha='left', va='top', transform=plt.gca().transAxes, color='blue')
plt.text(0.02, 0.91, f'Test MAE: {mae_test:.2f}, Test R²: {r2_test:.2f}', ha='left', va='top', transform=plt.gca().transAxes, color='red')


plt.savefig('density.png', dpi=300)
plt.show()


# 计算整个数据集的SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 选择显示的特征（第21到29列）
selected_shap_values = shap_values[:, 0:21]  # 选取第21到28列的SHAP值

# 选择对应的列（第21到29列）
selected_features = X_train.iloc[:, 0:21]

# 绘制SHAP值的汇总图，仅显示第21到29列特征
fig, ax = plt.subplots(figsize=(6, 4))  # 手动设置更窄的图像尺寸
shap.summary_plot(selected_shap_values, selected_features, show=False, max_display=20)
plt.savefig('shap_selected_features_Density.png', dpi=600)  # 保存图片
plt.show()