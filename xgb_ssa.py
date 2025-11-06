import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np

# 加载数据
data = pd.read_csv('递归式特征消除GSA20.csv')

# 分割数据为输入特征和输出特征
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 将数据分为训练集和测试集，90:10比例
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print("数据分割完成:")
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 对训练集进行5折交叉验证
print("\n开始5折交叉验证...")
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_mae = []
cv_scores_r2 = []

# 手动设置XGBoost超参数
params = {
    'n_estimators': 400,  # 树的数量
    'max_depth': 5,  # 树的最大深度
    'learning_rate': 0.01,  # 学习率
    'subsample': 0.8,  # 采样比例
    'colsample_bytree': 0.8,  # 特征采样比例
    'gamma': 0.3,  # 最小损失减少
    'reg_alpha': 3,  # L1正则
    'reg_lambda': 3,  # L2正则
    'random_state': 42,  # 随机种子
}

# 进行5折交叉验证
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train), 1):
    print(f"\n--- 第 {fold} 折交叉验证 ---")

    # 分割训练集和验证集
    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

    print(f"训练集大小: {X_train_fold.shape}, 验证集大小: {X_val_fold.shape}")

    # 训练模型
    fold_model = xgb.XGBRegressor(**params)
    fold_model.fit(
        X_train_fold,
        y_train_fold,
        eval_set=[(X_val_fold, y_val_fold)],
        verbose=False
    )

    # 在验证集上进行预测
    y_pred_val = fold_model.predict(X_val_fold)

    # 计算评估指标
    mae_val = mean_absolute_error(y_val_fold, y_pred_val)
    r2_val = r2_score(y_val_fold, y_pred_val)

    cv_scores_mae.append(mae_val)
    cv_scores_r2.append(r2_val)

    print(f"第 {fold} 折结果 - MAE: {mae_val:.4f}, R²: {r2_val:.4f}")

# 输出交叉验证结果
print("\n" + "=" * 50)
print("5折交叉验证结果:")
print(f"MAE 平均值: {np.mean(cv_scores_mae):.4f} (±{np.std(cv_scores_mae):.4f})")
print(f"R² 平均值: {np.mean(cv_scores_r2):.4f} (±{np.std(cv_scores_r2):.4f})")
print("\n各折详细结果:")
for i, (mae, r2) in enumerate(zip(cv_scores_mae, cv_scores_r2), 1):
    print(f"折 {i}: MAE = {mae:.4f}, R² = {r2:.4f}")

print("\n" + "=" * 50)
print("开始最终模型训练...")

# 使用全部训练数据训练最终模型
final_model = xgb.XGBRegressor(**params)
final_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

# 进行最终预测
y_pred_train = final_model.predict(X_train)
y_pred_test = final_model.predict(X_test)

# 评估最终模型
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f'\n最终模型结果:')
print(f'训练集 - Mean Absolute Error: {mae_train:.4f}, R²: {r2_train:.4f}')
print(f'测试集 - Mean Absolute Error: {mae_test:.4f}, R²: {r2_test:.4f}')

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

# 添加交叉验证结果
plt.text(0.02, 0.98, f'5-Fold CV MAE: {np.mean(cv_scores_mae):.2f} (±{np.std(cv_scores_mae):.2f})',
         ha='left', va='top', transform=plt.gca().transAxes, color='green')
plt.text(0.02, 0.93, f'5-Fold CV R²: {np.mean(cv_scores_r2):.2f} (±{np.std(cv_scores_r2):.2f})',
         ha='left', va='top', transform=plt.gca().transAxes, color='green')

plt.legend()
plt.savefig('xgb_SSA_with_cv.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存最终模型
joblib.dump(final_model, 'best_xgboost_model_with_cv.pkl')
print("\n模型已保存为 'best_xgboost_model_with_cv.pkl'")