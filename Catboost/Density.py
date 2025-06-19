import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 加载数据
data = pd.read_csv('递归式特征消除density20.csv', encoding='gbk')

# 分割数据为输入特征和输出特征
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

# 将数据分为训练集和测试集，90:10比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 定义类别特征的索引
cat_features = []

# 创建CatBoost Pool（包含类别特征的索引）
train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)

# 使用提供的最佳参数初始化CatBoost回归器
model = CatBoostRegressor(
    iterations=180,
    learning_rate=0.1,  # 可调整，通常0.1是一个好的起点
    depth=6,
    l2_leaf_reg=7,
    loss_function='RMSE',
    verbose=100
)

# 训练模型
model.fit(train_pool)

# 进行预测
y_pred_train = model.predict(train_pool)
y_pred_test = model.predict(test_pool)

# 只保留预测值中的正值
y_pred_train = [max(0, y) for y in y_pred_train]
y_pred_test = [max(0, y) for y in y_pred_test]

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

# 添加评估指标文本到左上角
plt.text(0.02, 0.97, f'Train MAE: {mae_train:.2f}, Train R²: {r2_train:.2f}', ha='left', va='top', transform=plt.gca().transAxes, color='blue')
plt.text(0.02, 0.91, f'Test MAE: {mae_test:.2f}, Test R²: {r2_test:.2f}', ha='left', va='top', transform=plt.gca().transAxes, color='red')

# 去掉图例
plt.legend().remove()

# 保存和显示图像
plt.savefig('Density20.png', dpi=300)
plt.show()

# 计算SHAP值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# 计算SHAP特征重要性总和
shap_importance = np.abs(shap_values).mean(axis=0)

# 计算每个特征的重要性比例并转换为百分比
importance_ratio = shap_importance / shap_importance.sum() * 100  # 转换为百分比

# 将特征重要性比例转换为 DataFrame 以便查看
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': shap_importance,
    'Importance Ratio (%)': importance_ratio
})

# 按重要性比例排序
importance_df = importance_df.sort_values(by='Importance Ratio (%)', ascending=False)

# 输出结果
print("Feature Importance and its Ratio (in percentage):")
print(importance_df)

# 可选：将结果保存到 CSV 文件
importance_df.to_csv('feature_importance_ratio_percent.csv', index=False)

# 绘制SHAP特征重要性图
plt.figure(figsize=(6, 6))  # 更大的图像尺寸
shap.summary_plot(shap_values, X_train, plot_type="bar", show=False, max_display=10)
plt.savefig('shap_feature_importance_Density20.png', dpi=600)  # 更高的分辨率
plt.show()

# 绘制SHAP值的汇总图，控制最大显示的特征数量
plt.figure(figsize=(6, 6))  # 更大的图像尺寸
shap.summary_plot(shap_values, X_train, show=False, max_display=10)  # 显示前15个重要特征
plt.savefig('shap_Density20_M+L.png', dpi=600)  # 更高的分辨率
plt.show()