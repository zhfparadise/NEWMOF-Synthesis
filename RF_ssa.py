import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer  # 导入缺失值处理工具
import joblib

# 加载数据
data = pd.read_csv('递归式特征消除GSA20.csv')

# 分割数据为输入特征和输出特征
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 检查数据中是否有缺失值
print("缺失值统计:")
print(X.isnull().sum())

# 处理缺失值 - 使用中位数填充
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# 将数据分为训练集和测试集，90:10比例
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.1, random_state=42)

# 设置随机森林超参数
params = {
    'n_estimators': 600,         # 树的数量
    'max_depth': 9,              # 树的最大深度
    'min_samples_split': 2,      # 分裂所需的最小样本数
    'min_samples_leaf': 1,       # 叶节点所需的最小样本数
    'max_features': 'sqrt',      # 分裂时考虑的特征数量 ('sqrt'表示特征总数的平方根)
    'bootstrap': True,           # 是否使用bootstrap抽样
    'random_state': 42,          # 随机种子
}

# 训练模型
model = RandomForestRegressor(**params)

# 在训练集上进行5折交叉验证
print("正在进行5折交叉验证...")
scoring = {'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
cv_results = cross_validate(model, X_train, y_train, cv=5,
                           scoring=scoring, return_train_score=True)

# 输出交叉验证结果
print("\n=== 5折交叉验证结果 ===")
print("训练集指标:")
print(f"MAE: {-cv_results['train_mae'].mean():.4f} (+/- {-cv_results['train_mae'].std() * 2:.4f})")
print(f"R²: {cv_results['train_r2'].mean():.4f} (+/- {cv_results['train_r2'].std() * 2:.4f})")

print("\n验证集指标:")
print(f"MAE: {-cv_results['test_mae'].mean():.4f} (+/- {-cv_results['test_mae'].std() * 2:.4f})")
print(f"R²: {cv_results['test_r2'].mean():.4f} (+/- {cv_results['test_r2'].std() * 2:.4f})")

# 输出每一折的详细结果
print("\n=== 详细分折结果 ===")
for i in range(5):
    print(f"折 {i+1}:")
    print(f"  训练集 - MAE: {-cv_results['train_mae'][i]:.4f}, R²: {cv_results['train_r2'][i]:.4f}")
    print(f"  验证集 - MAE: {-cv_results['test_mae'][i]:.4f}, R²: {cv_results['test_r2'][i]:.4f}")

# 使用全部训练数据重新训练最终模型
print("\n训练最终模型...")
model.fit(X_train, y_train)

# 进行预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 评估最终模型
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'\n=== 最终模型在测试集上的表现 ===')
print(f'Mean Absolute Error on train set: {mae_train:.4f}, R² on train set: {r2_train:.4f}')
print(f'Mean Absolute Error on test set: {mae_test:.4f}, R² on test set: {r2_test:.4f}')

# 确定绘制参考线的范围
min_val = min(y_train.min(), y_test.min())
max_val = max(y_train.max(), y_test.max())

# 设置字体大小
plt.rcParams.update({'font.size': 17})

# 绘制实际值与预测值的散点图
plt.figure(figsize=(8, 8))
plt.scatter(y_train, y_pred_train, alpha=0.5, color='blue', label='Train')
plt.scatter(y_test, y_pred_test, alpha=0.5, color='red', label='Test')
plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # 画y=x的参考线
plt.xlabel('Actual SSA [m²/g]')
plt.ylabel('Predicted SSA [m²/g]')

# 添加评估指标文本
plt.text(0.98, 0.06, f'Train MAE: {mae_train:.2f}, R²: {r2_train:.2f}', ha='right', va='bottom',
         transform=plt.gca().transAxes, color='blue')
plt.text(0.98, 0.01, f'Test MAE: {mae_test:.2f}, R²: {r2_test:.2f}', ha='right', va='bottom',
         transform=plt.gca().transAxes, color='red')

plt.legend()
plt.savefig('rf_SSA.png', dpi=300)  # 使用不同的文件名
plt.show()

# 保存模型
joblib.dump(model, 'best_random_forest_model.pkl')

# 特征重要性分析
feature_importances = model.feature_importances_
features = X.columns  # 使用原始的特征名称

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# 按重要性排序
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# 打印最重要的10个特征
print("\nTop 10 important features:")
print(importance_df.head(10))

# 绘制特征重要性图
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:15], importance_df['Importance'][:15])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Top 15 Feature Importances')
plt.gca().invert_yaxis()  # 最重要的特征显示在顶部
plt.tight_layout()
plt.savefig('rf_feature_importances.png', dpi=300)
plt.show()