from scipy.spatial import cKDTree
import numpy as np
import pandas as pd


# 加载表1和表2
best_solutions = pd.read_csv('pareto_optimal_solutions.csv', encoding='gbk')
solvent_types = pd.read_csv('solvent type.csv', encoding='gbk')
metal_types = pd.read_csv('Metal type.csv', encoding='gbk')
descriptors = pd.read_csv('descriptors1.csv', encoding='gbk')

# 找出表1、表2、金属类型和描述符的重复列
common_columns_solvent = best_solutions.columns.intersection(solvent_types.columns)
common_columns_metal = best_solutions.columns.intersection(metal_types.columns)
common_columns_descriptors = best_solutions.columns.intersection(descriptors.columns)

# 只保留表2、金属类型和描述符中与表1重复的列
solvent_types_filtered = solvent_types[common_columns_solvent]
metal_types_filtered = metal_types[common_columns_metal]
descriptors_filtered = descriptors[common_columns_descriptors]

# 提取表1、表2、金属类型和描述符中的重复列值
best_solutions_filtered_solvent = best_solutions[common_columns_solvent].values
solvent_values = solvent_types[common_columns_solvent].values
solvent_first_column = solvent_types.iloc[:, 0].values  # 表2的第一列

best_solutions_filtered_metal = best_solutions[common_columns_metal].values
metal_values = metal_types[common_columns_metal].values
metal_first_column = metal_types.iloc[:, 0].values  # 金属类型的第一列

best_solutions_filtered_descriptors = best_solutions[common_columns_descriptors].values
descriptor_values = descriptors[common_columns_descriptors].values
descriptor_first_column = descriptors.iloc[:, 0].values  # 描述符的第一列

# 使用cKDTree寻找最近的溶剂值
tree_solvent = cKDTree(solvent_values)
distances_solvent, indices_solvent = tree_solvent.query(best_solutions_filtered_solvent)

# 将最接近的表2的第一列对应值与表1合并
best_solutions['Matched_Solvent_Type'] = solvent_first_column[indices_solvent]

# 使用cKDTree寻找最近的金属值
tree_metal = cKDTree(metal_values)
distances_metal, indices_metal = tree_metal.query(best_solutions_filtered_metal)

# 将最接近的金属类型的第一列对应值与表1合并
best_solutions['Matched_Metal_Type'] = metal_first_column[indices_metal]

# 使用cKDTree寻找最近的描述符值
tree_descriptors = cKDTree(descriptor_values)
distances_descriptors, indices_descriptors = tree_descriptors.query(best_solutions_filtered_descriptors)

# 将最接近的描述符的第一列对应值与表1合并
best_solutions['Matched_Descriptor'] = descriptor_first_column[indices_descriptors]

# 输出匹配结果
print(best_solutions[['Matched_Solvent_Type', 'Matched_Metal_Type', 'Matched_Descriptor']].head())

# 保存匹配结果到新表
best_solutions[['Matched_Solvent_Type', 'Matched_Metal_Type', 'Matched_Descriptor']].to_csv('matched_results1.csv', index=False)

# ===== 修正后的准确度计算 =====
# 假设主表包含真实标签列（根据实际列名修改）
SOLVENT_TRUTH_COL = 'Solvent_Type'  # 主表中溶剂真实标签的列名
METAL_TRUTH_COL = 'Metal_Type'      # 主表中金属真实标签的列名
DESCRIPTOR_TRUTH_COL = 'Descriptor' # 主表中描述符真实标签的列名

# 确保主表包含真实标签列
if SOLVENT_TRUTH_COL in best_solutions.columns:
    # 计算溶剂匹配准确度
    correct_matches_solvent = (best_solutions['Matched_Solvent_Type'] == best_solutions[SOLVENT_TRUTH_COL])
    accuracy_solvent = np.mean(correct_matches_solvent) * 100
    print(f"Matching Accuracy for Solvent: {accuracy_solvent:.2f}%")
else:
    print(f"Warning: Solvent truth column '{SOLVENT_TRUTH_COL}' not found in best_solutions")

if METAL_TRUTH_COL in best_solutions.columns:
    # 计算金属匹配准确度
    correct_matches_metal = (best_solutions['Matched_Metal_Type'] == best_solutions[METAL_TRUTH_COL])
    accuracy_metal = np.mean(correct_matches_metal) * 100
    print(f"Matching Accuracy for Metal: {accuracy_metal:.2f}%")
else:
    print(f"Warning: Metal truth column '{METAL_TRUTH_COL}' not found in best_solutions")

if DESCRIPTOR_TRUTH_COL in best_solutions.columns:
    # 计算描述符匹配准确度
    correct_matches_descriptors = (best_solutions['Matched_Descriptor'] == best_solutions[DESCRIPTOR_TRUTH_COL])
    accuracy_descriptors = np.mean(correct_matches_descriptors) * 100
    print(f"Matching Accuracy for Descriptors: {accuracy_descriptors:.2f}%")
else:
    print(f"Warning: Descriptor truth column '{DESCRIPTOR_TRUTH_COL}' not found in best_solutions")