from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

# 加载表1和表2
best_solutions = pd.read_csv('pareto_optimal_solutions_pv.csv')
solvent_types = pd.read_csv('solvent type.csv')
metal_types = pd.read_csv('Metal type.csv')
descriptors = pd.read_csv('descriptors1.csv')

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
best_solutions[['Matched_Solvent_Type', 'Matched_Metal_Type', 'Matched_Descriptor']].to_csv('pv_matched_results.csv', index=False)

# 计算溶剂匹配准确度
actual_solvent_types = solvent_types.iloc[:, 0].values
matched_solvent_types = solvent_first_column[indices_solvent]
correct_matches_solvent = matched_solvent_types == actual_solvent_types[indices_solvent]
accuracy_solvent = np.sum(correct_matches_solvent) / len(correct_matches_solvent) * 100

# 计算金属匹配准确度
actual_metal_types = metal_types.iloc[:, 0].values
matched_metal_types = metal_first_column[indices_metal]
correct_matches_metal = matched_metal_types == actual_metal_types[indices_metal]
accuracy_metal = np.sum(correct_matches_metal) / len(correct_matches_metal) * 100

# 计算描述符匹配准确度
actual_descriptors = descriptors.iloc[:, 0].values
matched_descriptors = descriptor_first_column[indices_descriptors]
correct_matches_descriptors = matched_descriptors == actual_descriptors[indices_descriptors]
accuracy_descriptors = np.sum(correct_matches_descriptors) / len(correct_matches_descriptors) * 100

# 打印准确度
print(f"Matching Accuracy for Solvent: {accuracy_solvent:.2f}%")
print(f"Matching Accuracy for Metal: {accuracy_metal:.2f}%")
print(f"Matching Accuracy for Descriptors: {accuracy_descriptors:.2f}%")
