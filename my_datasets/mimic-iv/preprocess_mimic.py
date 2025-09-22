import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from my_datasets.utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time, export_note

processed_data_dir = os.path.join("./my_datasets/mimic-iv", 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Record feature names
basic_records = ['RecordID', 'PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
note_features = ['Text']
demographic_features = ['Sex', 'Age']
labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']


# =================================================================================================
# 1. 读取和初始预处理 (在划分数据集之前完成)
# =================================================================================================

# 读取数据集
print("Reading parquet file...")
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic-iv-timeseries-note.parquet'))
df = df[basic_records + target_features + note_features + demographic_features + labtest_features]

# Ensure the data is sorted by RecordID and RecordTime
df = df.sort_values(by=['RecordID', 'RecordTime']).reset_index(drop=True)

# --- 导出在one-hot编码之前的原始信息 ---
# 这些信息不应被标准化或one-hot编码，用于LLM等模型
print("Exporting raw information (missing mask, record time, raw features)...")
raw_missing_mask_map = {id: mask for id, mask in zip(df['RecordID'].unique(), export_missing_mask(df, demographic_features, labtest_features, id_column='RecordID'))}
raw_record_time_map = {id: time for id, time in zip(df['RecordID'].unique(), export_record_time(df, id_column='RecordID'))}
_, raw_x_list, _, raw_pid_list = forward_fill_pipeline(df.copy(), None, demographic_features, labtest_features, target_features, [], id_column='RecordID')
raw_x_map = {id: x for id, x in zip(raw_pid_list, raw_x_list)}


# --- 对分类特征进行One-Hot编码 ---
print("Performing one-hot encoding...")
one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
columns = df.columns.to_list()
column_start_idx = columns.index(categorical_labtest_features[0])
column_end_idx = columns.index(categorical_labtest_features[-1])
df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)

# 更新特征列表
ehr_categorical_labtest_features = one_hot.columns.to_list()
ehr_labtest_features = ehr_categorical_labtest_features + numerical_labtest_features
require_impute_features = ehr_labtest_features


# =================================================================================================
# 2. 数据集划分 (先划出测试集，再从剩余数据中采样训练集和验证集)
# =================================================================================================
print("Splitting dataset...")
# 按 RecordID 分组，并获取患者ID和对应的结局用于分层抽样
grouped = df.groupby('RecordID')
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# 随机分出200个患者作为固定的测试集
train_val_patients, test_patients, train_val_outcomes, _ = train_test_split(
    patients, patients_outcome, test_size=200, random_state=SEED, stratify=patients_outcome
)

# 定义训练集和验证集的大小
train_sizes = [20, 50, 100, 200, 400, 800, 1600, 3200, 6400]
val_sizes = train_sizes # 验证集与训练集大小一致

# 循环创建不同大小的数据集
for i, (train_size, val_size) in enumerate(zip(train_sizes, val_sizes)):
    print("-" * 50)
    print(f"Processing split {i+1}: train_size={train_size}, val_size={val_size}")

    # --- 从剩余患者中不放回地采样训练集和验证集 ---
    # 确保总样本足够
    if train_size + val_size > len(train_val_patients):
        print(f"Warning: Not enough patients for train_size={train_size} and val_size={val_size}. Skipping.")
        continue

    # 1. 采样训练集
    train_patients, remaining_patients, train_outcomes, remaining_outcomes = train_test_split(
        train_val_patients, train_val_outcomes, train_size=train_size, random_state=SEED + i, stratify=train_val_outcomes
    )

    # 2. 从剩余部分采样验证集
    # 如果剩余样本不足以进行分层抽样，则进行随机抽样
    try:
        val_patients, _, _, _ = train_test_split(
            remaining_patients, remaining_outcomes, train_size=val_size, random_state=SEED + i, stratify=remaining_outcomes
        )
    except ValueError:
        print(f"Warning: Could not stratify validation set for size {val_size}. Using random sampling.")
        val_indices = np.random.choice(len(remaining_patients), size=val_size, replace=False)
        val_patients = remaining_patients[val_indices]

    print(f"Train patients size: {len(train_patients)}")
    print(f"Validation patients size: {len(val_patients)}")
    print(f"Test patients size: {len(test_patients)}")

    # 断言检查数据泄漏
    assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage between train and val sets"
    assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage between train and test sets"
    assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage between val and test sets"

    # --- 根据划分的ID创建DataFrame ---
    train_df_raw = df[df['RecordID'].isin(train_patients)].copy()
    val_df_raw = df[df['RecordID'].isin(val_patients)].copy()
    test_df_raw = df[df['RecordID'].isin(test_patients)].copy()

    # =================================================================================================
    # 3. 标准化与填充 (使用当前训练集的统计数据)
    # =================================================================================================
    print("Normalizing and imputing data...")

    # 计算训练集的均值和标准差，并应用到所有数据集
    train_df, val_df, test_df, default_fill, los_info, _, _ = normalize_dataframe(
        train_df_raw, val_df_raw, test_df_raw, normalize_features, id_column="RecordID"
    )

    # 正向填充
    _, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
    _, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
    _, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")

    # 提取文本笔记
    train_note = export_note(train_df, id_column='RecordID')
    val_note = export_note(val_df, id_column='RecordID')
    test_note = export_note(test_df, id_column='RecordID')

    # =================================================================================================
    # 4. 数据格式化与保存
    # =================================================================================================

    # 创建保存目录
    save_dir = os.path.join(processed_data_dir, f'{train_size}_shot')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving processed data to {save_dir}")

    # 定义数据转换函数，避免代码重复
    def format_data(pids, x_ts, y_ts, notes, raw_x_map, time_map, mask_map):
        data = []
        for i, pid in enumerate(pids):
            data.append({
                'id': pid,
                'x_ts': x_ts[i],
                'x_note': notes[i],
                'x_llm_ts': raw_x_map.get(pid, []), # 从map中获取原始数据
                'record_time': time_map.get(pid, []),
                'missing_mask': mask_map.get(pid, []),
                'y_mortality': [y[0] for y in y_ts[i]],
                'y_los': [y[1] for y in y_ts[i]],
                'y_readmission': [y[2] for y in y_ts[i]],
            })
        return data

    # 格式化数据
    train_data = format_data(train_pid, train_x, train_y, train_note, raw_x_map, raw_record_time_map, raw_missing_mask_map)
    val_data = format_data(val_pid, val_x, val_y, val_note, raw_x_map, raw_record_time_map, raw_missing_mask_map)
    test_data = format_data(test_pid, test_x, test_y, test_note, raw_x_map, raw_record_time_map, raw_missing_mask_map)

    # 保存为 pickle 文件
    pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
    pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
    pd.to_pickle(test_data, os.path.join(save_dir, "test_data.pkl"))

    print(f"Saved data sizes: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

    # 保存 LOS 统计信息和特征列表 (对于每个子集都保存一份，或者也可以只保存一份在根目录)
    pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))
    pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))

print("-" * 50)
print("All processing finished.")