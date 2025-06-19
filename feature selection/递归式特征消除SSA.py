import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # 导入填充缺失值的工具

# 读取数据
file_path = r"SSA690.csv"
data = pd.read_csv(file_path, encoding='gbk')
X = data.iloc[:, 4:-1]
y = data.iloc[:, -1]
print(X)
print(y)

# 使用 SimpleImputer 填充缺失值（使用均值填充）
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 使用随机森林回归器
rfr = RandomForestRegressor(n_estimators=200, random_state=0)

# 递归特征消除
rfe = RFE(estimator=rfr, n_features_to_select=20, step=1)

# 进行特征选择
X_RFE = rfe.fit_transform(X_imputed, y)
print(X_RFE.shape)

# 获取选中特征的列名
selected_columns = X.columns[rfe.get_support()]

# 创建新的 DataFrame 包含选中特征和目标列
new_data = pd.DataFrame(X_RFE, columns=selected_columns)
new_data['Target'] = y.values

# 打印新的 DataFrame
print(new_data)

# 将新的 DataFrame 导出为 CSV 文件
new_file_path = r'递归式特征消除SSA20.csv'
new_data.to_csv(new_file_path, index=False)
print(f"新的特征数据已保存至 {new_file_path}")
