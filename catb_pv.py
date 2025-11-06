import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import os
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.interpolate import make_interp_spline

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.family'] = 'Arial'

# 加载数据
data = pd.read_csv('递归式特征消除POV_重量20.csv')
X = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 保留10%作为最终测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 初始化5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 存储每折的结果
cv_mae_scores = []
cv_r2_scores = []
models = []  # 存储每个fold的模型

print("开始5折交叉验证...")

# 5折交叉验证
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
    print(f"\n=== 第 {fold} 折 ===")

    # 划分训练集和验证集
    X_train, X_val = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # 类别特征（可选）
    cat_features = []

    # 构建CatBoost Pool
    train_pool = Pool(data=X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(data=X_val, label=y_val, cat_features=cat_features)

    # 初始化模型
    model = CatBoostRegressor(
        iterations=110,
        learning_rate=0.1,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        verbose=100
    )
    model.fit(train_pool, eval_set=val_pool, verbose=100)

    # 预测
    y_pred_train = [max(0, y) for y in model.predict(train_pool)]
    y_pred_val = [max(0, y) for y in model.predict(val_pool)]

    # 评估
    mae_train = mean_absolute_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    r2_val = r2_score(y_val, y_pred_val)

    print(f'训练集 - MAE: {mae_train:.4f}, R²: {r2_train:.4f}')
    print(f'验证集 - MAE: {mae_val:.4f}, R²: {r2_val:.4f}')

    cv_mae_scores.append(mae_val)
    cv_r2_scores.append(r2_val)
    models.append(model)

# 输出交叉验证结果
print("\n=== 5折交叉验证结果 ===")
print(f'MAE 平均值: {np.mean(cv_mae_scores):.4f} (±{np.std(cv_mae_scores):.4f})')
print(f'R² 平均值: {np.mean(cv_r2_scores):.4f} (±{np.std(cv_r2_scores):.4f})')

# 选择最佳模型（基于验证集R²）
best_fold = np.argmax(cv_r2_scores)
best_model = models[best_fold]
print(f"\n最佳模型: 第 {best_fold + 1} 折 (R² = {cv_r2_scores[best_fold]:.4f})")

# 使用最佳模型在最终测试集上进行评估
test_pool = Pool(data=X_test, label=y_test, cat_features=cat_features)
y_pred_test = [max(0, y) for y in best_model.predict(test_pool)]
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n=== 最终测试集结果 (使用最佳模型) ===")
print(f'测试集 - MAE: {mae_test:.4f}, R²: {r2_test:.4f}')

# 使用整个训练集重新训练最终模型
print("\n=== 使用整个训练集训练最终模型 ===")
final_train_pool = Pool(data=X_train_full, label=y_train_full, cat_features=cat_features)
final_model = CatBoostRegressor(
    iterations=110,
    learning_rate=0.1,
    depth=6,
    l2_leaf_reg=3,
    loss_function='RMSE',
    verbose=100
)
final_model.fit(final_train_pool)

# 在最终测试集上评估
y_pred_final_test = [max(0, y) for y in final_model.predict(test_pool)]
mae_final_test = mean_absolute_error(y_test, y_pred_final_test)
r2_final_test = r2_score(y_test, y_pred_final_test)

print(f"\n=== 最终模型在测试集上的表现 ===")
print(f'测试集 - MAE: {mae_final_test:.4f}, R²: {r2_final_test:.4f}')

# 绘制实际值与预测值对比图（使用最终模型在整个训练集和测试集上的预测）
y_pred_final_train = [max(0, y) for y in final_model.predict(final_train_pool)]

plt.figure(figsize=(6, 6))
plt.scatter(y_train_full, y_pred_final_train, alpha=0.9, color='#196DDE', label='Train', s=75)
plt.scatter(y_test, y_pred_final_test, alpha=0.9, color='#E91313', label='Test', s=75)
plt.plot([min(y_train_full.min(), y_test.min()), max(y_train_full.max(), y_test.max())],
         [min(y_train_full.min(), y_test.min()), max(y_train_full.max(), y_test.max())], 'k--')
plt.xlabel('Actual PV [cm³/g]', fontsize=22)
plt.ylabel('Predicted PV [cm³/g]', fontsize=22)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.text(0.02, 0.86,
         f'Train R²: {r2_score(y_train_full, y_pred_final_train):.2f}\nMAE: {mean_absolute_error(y_train_full, y_pred_final_train):.2f}',
         transform=plt.gca().transAxes, alpha=1, color='#196DDE', fontsize=22)
plt.text(0.02, 0.72, f'Test R²: {r2_final_test:.2f}\nMAE: {mae_final_test:.2f}',
         transform=plt.gca().transAxes, alpha=1, color='#E91313', fontsize=22)
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()
plt.savefig('actual_vs_predicted_pv.png', dpi=600, bbox_inches='tight')
plt.show()

# 保存最终模型
final_model.save_model('best_catboost_model.cbm')
joblib.dump(final_model, 'model_pv.pkl')

# 计算SHAP值（使用最终模型）
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_full)

# 设置全局字体大小
plt.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.size': 16})

# 绘制SHAP特征重要性图 (条形图)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_full, plot_type="bar", show=False, max_display=18)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=16)
plt.savefig('shap_feature_importance_PV.png', dpi=600, bbox_inches='tight')
plt.show()

# 绘制SHAP值的汇总图 (beeswarm图)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_train_full, show=False, max_display=18)
ax = plt.gca()
ax.tick_params(axis='both', labelsize=16)
ax.set_xlabel("SHAP value", fontsize=16)

# 修改颜色条的字体大小
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)
cbar.set_ylabel('Feature value', fontsize=16)

for text in cbar.yaxis.get_ticklabels():
    text.set_fontsize(16)

plt.savefig('shap_PV.png', dpi=600, bbox_inches='tight')
plt.show()

# 修复：定义selected_shap_values和selected_features
if X_train_full.shape[1] >= 35:
    selected_shap_values = shap_values[:, 28:35]
    selected_features = X_train_full.iloc[:, 28:35]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(selected_shap_values, selected_features, show=False, max_display=20)
    ax = plt.gca()
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel("SHAP value", fontsize=18)

    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=18)
    cbar.set_ylabel('Feature value', fontsize=18)

    for text in cbar.yaxis.get_ticklabels():
        text.set_fontsize(18)

    plt.savefig('shap_selected_features_PV.png', dpi=600, bbox_inches='tight')
    plt.show()
else:
    print(f"警告：数据集只有 {X_train_full.shape[1]} 个特征，无法选取第29-35列")

# 恢复默认字体设置
plt.rcParams.update(plt.rcParamsDefault)
mpl.rcParams.update(mpl.rcParamsDefault)