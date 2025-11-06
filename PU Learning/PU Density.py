import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ========== 1. 读取数据 ==========
P_df = pd.read_csv('多目标density.csv')  # 全部正例数据
PV_df = pd.read_csv('density_best_solutions.csv')  # 待测试数据

# ========== 2. 删除无变异特征（同时从 SSA_df 中删除）==========
no_var_cols = P_df.loc[:, P_df.std() == 0].columns.tolist()
if no_var_cols:
    print('删除无变异特征列:', no_var_cols)
    P_df.drop(columns=no_var_cols, inplace=True)
    PV_df.drop(columns=[col for col in no_var_cols if col in PV_df.columns], inplace=True)

# ========== 3. 填补 SSA_df 缺失列，并统一列顺序 ==========
for col in P_df.columns:
    if col not in PV_df.columns:
        PV_df[col] = P_df[col].median()  # 用正例列的中位数填补
SSA_df = PV_df[P_df.columns]  # 按正例列顺序排列

# ========== 4. 正例划分训练测试集 ==========
P_train, P_test = train_test_split(P_df, test_size=0.2, random_state=42)

# ========== 5. 特征归一化 ==========
scaler = MinMaxScaler()
P_train_scaled = scaler.fit_transform(P_train)
P_test_scaled = scaler.transform(P_test)
SSA_scaled = scaler.transform(SSA_df)

# ========== 6. 构造未标记样本 ==========
U_num = 10000
np.random.seed(42)
U_data_raw = P_train.sample(n=U_num, replace=True).values  # bootstrap 采样
noise = np.random.normal(loc=0.0, scale=0.1, size=U_data_raw.shape)
U_data = U_data_raw + noise
U_scaled = scaler.transform(U_data)

# ========== 7. 处理 NaN ==========
P_train_scaled = np.nan_to_num(P_train_scaled)
P_test_scaled = np.nan_to_num(P_test_scaled)
SSA_scaled = np.nan_to_num(SSA_scaled)
U_scaled = np.nan_to_num(U_scaled)

# ========== 8. 模型结构 ==========
def create_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='elu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

# ========== 9. Bagging PU 学习 ==========
N_iter = 30
PU_models = []

for i in range(N_iter):
    np.random.seed(i)
    # Bootstrap 正例
    idx = np.random.choice(len(P_train_scaled), size=len(P_train_scaled), replace=True)
    P_sampled = P_train_scaled[idx]
    U_sampled = U_scaled

    X_train = np.vstack([P_sampled, U_sampled])
    y_train = np.hstack([np.ones(len(P_sampled)), np.zeros(len(U_sampled))])

    model = create_model(P_sampled.shape[1])
    model.fit(X_train, y_train, epochs=100, batch_size=128, verbose=0)
    PU_models.append(model)

# ========== 10. 预测函数 ==========
def predict_crystal_score(models, X):
    preds = np.array([m.predict(X, verbose=0).flatten() for m in models])
    return preds.mean(axis=0)

# ========== 11. 模型预测 ==========
P_test_score = predict_crystal_score(PU_models, P_test_scaled)
U_score = predict_crystal_score(PU_models, U_scaled)
SSA_score = predict_crystal_score(PU_models, SSA_scaled)

# ========== 12. 可视化 ==========

# 设置全局字体大小（也可设置为 font.family）
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 22
})


sns.set(style='whitegrid', font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'  # 避免中文乱码时改为合适字体
plt.rcParams['axes.unicode_minus'] = False

def plot_score_distribution(scores, threshold, color, xlabel, title, filename, extra_text=None):
    plt.figure(figsize=(8, 5))
    sns.histplot(scores, bins=50, color=color, kde=True, alpha=0.7, edgecolor='black')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
    if extra_text:
        plt.text(threshold + 0.02, plt.ylim()[1] * 0.9, extra_text, color='red')

    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# 图1：正例测试集得分分布
recall = (P_test_score > 0.5).sum() / len(P_test_score)
print(f"正例测试集召回率: {recall:.4f}")
plot_score_distribution(
    scores=P_test_score,
    threshold=0.5,
    color='seagreen',
    xlabel='Crystal Score',
    title=f'Positive Test Score Distribution\nRecall = {recall:.2%}',
    filename='crystal_score_P_test_Density.png',
    extra_text=f"Recall: {recall:.2%}"
)

# 图2：未标记数据得分分布
U_positive_rate = (U_score > 0.5).sum() / len(U_score)
print(f"未标记数据中预测为正例的比例: {U_positive_rate:.4f}")
plot_score_distribution(
    scores=U_score,
    threshold=0.5,
    color='slategray',
    xlabel='Crystal Score',
    title=f'Unlabeled Score Distribution\nPositive Rate = {U_positive_rate:.2%}',
    filename='crystal_score_U_Density.png',
    extra_text=f"Pos Rate: {U_positive_rate:.2%}"
)

# 图3：SSA最佳解集得分分布
plot_score_distribution(
    scores=SSA_score,
    threshold=0.5,
    color='steelblue',
    xlabel='Crystal Score',
    title='Density Best Set Score Distribution',
    filename='crystal_score_Density.png'
)


# ========== 13. 保存结果 ==========
pd.DataFrame({'Positive_Test_Score': P_test_score}).to_csv('P_test_score.csv', index=False)
pd.DataFrame({'Unlabeled_Score': U_score}).to_csv('U_score.csv', index=False)
pd.DataFrame({'Density_Best_Score': SSA_score}).to_csv('Density_score.csv', index=False)

# ========== 14. 保存模型 ==========
os.makedirs('PU_models', exist_ok=True)
for i, model in enumerate(PU_models):
    model.save(f'PU_models/PU_model_{i}.keras')
