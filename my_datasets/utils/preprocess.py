from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


def calculate_data_existing_length(data: np.ndarray) -> int:
    """
    计算数据中非缺失值(非NaN)的数量。

    Args:
        data: 输入数据数组

    Returns:
        非NaN值的数量
    """
    return sum(1 for item in data if not pd.isna(item))


def fill_missing_value(data: np.ndarray, to_fill_value: float = 0) -> np.ndarray:
    """
    填充时间序列数据中的缺失值。数据按时间升序排列。
    对于序列开始的缺失值使用指定的填充值，之后的缺失值使用前向填充策略。

    Args:
        data: 按时间升序排列的数据数组
        to_fill_value: 用于填充序列开始部分缺失值的默认值

    Returns:
        填充后的数据数组
    """
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)

    # 如果没有缺失值，直接返回原数据
    if data_len == data_exist_len:
        return data

    # 如果全是缺失值，全部填充为指定值
    if data_exist_len == 0:
        data[:] = to_fill_value
        return data

    # 处理序列开始部分的缺失值
    if pd.isna(data[0]):
        # 查找第一个非NaN值的位置
        not_na_pos = next((i for i, val in enumerate(data) if not pd.isna(val)), 0)
        # 填充第一个非NaN值之前的元素
        data[:not_na_pos] = to_fill_value

    # 使用前向填充策略处理剩余的缺失值
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]

    return data


def forward_fill_pipeline(
    df: pd.DataFrame,
    default_fill: pd.DataFrame,
    demographic_features: List[str],
    labtest_features: List[str],
    target_features: List[str],
    require_impute_features: List[str],
    id_column: str="PatientID"
) -> Tuple[pd.DataFrame, List, List, List]:
    """
    针对患者数据的前向填充管道。对每个患者的时间序列数据进行排序，填充缺失值，
    并提取特征和目标变量。

    Args:
        df: 包含患者数据的DataFrame
        default_fill: 默认填充值的DataFrame
        demographic_features: 人口统计学特征列表
        labtest_features: 实验室检测特征列表
        target_features: 目标变量特征列表
        require_impute_features: 需要进行缺失值填充的特征列表

    Returns:
        元组，包含(
            填充后的DataFrame,
            所有患者的特征列表,
            所有患者的目标变量列表,
            所有患者ID列表
        )
    """
    df_copy = df.copy()
    grouped = df_copy.groupby(id_column)
    all_x, all_y, all_pid = [], [], []

    for patient_id, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)

        # 对需要填充的特征进行缺失值处理
        for feature in require_impute_features:
            # 确定填充值，分类特征默认为-1
            to_fill_value = default_fill.get(feature, -1)
            # 使用患者中位数作为缺失值的填充值
            fill_missing_value(sorted_group[feature].values, to_fill_value)

        # 提取特征和目标变量
        patient_x = []
        patient_y = []

        for _, row in sorted_group.iterrows():
            # 提取目标变量
            target_values = [row[f] for f in target_features]
            patient_y.append(target_values)

            # 提取特征
            features = [row[f] for f in demographic_features + labtest_features]
            patient_x.append(features)

        all_x.append(patient_x)
        all_y.append(patient_y)
        all_pid.append(patient_id)

    return df_copy, all_x, all_y, all_pid


def export_missing_mask(
    df: pd.DataFrame,
    demographic_features: List[str],
    labtest_features: List[str],
    id_column: str = "PatientID",
) -> List:
    """
    导出缺失值掩码，标记每个特征的缺失值。

    Args:
        df: 包含患者数据的DataFrame
        demographic_features: 人口统计学特征列表
        labtest_features: 实验室检测特征列表

    Returns:
        缺失值掩码列表
    """
    grouped = df.groupby(id_column)
    missing_mask = []

    for _, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)

        # 创建所有特征的缺失值掩码
        features = demographic_features + labtest_features
        patient_mask = []

        for _, row in sorted_group.iterrows():
            feature_mask = [1 if pd.isna(row[f]) else 0 for f in features]
            patient_mask.append(feature_mask)

        missing_mask.append(patient_mask)

    return missing_mask


def export_record_time(
    df: pd.DataFrame,
    id_column: str = "PatientID",
) -> List:
    """
    导出记录时间序列，按患者ID分组。

    Args:
        df: 包含患者数据的DataFrame

    Returns:
        记录时间序列列表
    """
    grouped = df.groupby(id_column)
    record_time = []

    for _, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)
        record_time.append(sorted_group["RecordTime"].values.tolist())

    return record_time


def export_note(
    df: pd.DataFrame,
    id_column: str = "RecordID",
    note_column: str = "Text",
) -> List:
    """
    导出患者的临床笔记信息，按患者ID分组。

    Args:
        df: 包含患者数据的DataFrame

    Returns:
        临床笔记列表
    """
    grouped = df.groupby(id_column)
    notes = grouped.first()[note_column].tolist()

    return notes


def filter_outlier(element: float) -> float:
    """
    过滤异常值，将绝对值大于10000的值替换为0。

    Args:
        element: 输入值

    Returns:
        过滤后的值
    """
    return 0 if abs(float(element)) > 1e4 else element


def normalize_dataframe(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    normalize_features: List[str],
    id_column: str = "PatientID",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict[str, float]], pd.Series, pd.Series]:
    """
    对训练集、验证集和测试集的特征进行标准化处理，并计算相关统计信息。

    Args:
        train_df: 训练集DataFrame
        val_df: 验证集DataFrame
        test_df: 测试集DataFrame
        normalize_features: 需要标准化的特征列表

    Returns:
        元组，包含(
            标准化后的训练集,
            标准化后的验证集,
            标准化后的测试集,
            默认填充值DataFrame,
            住院时长(LOS)相关信息字典(如果存在),
            训练集均值,
            训练集标准差
        )
    """
    # 计算分位数，过滤掉异常值
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)
    filtered_df = train_df[
        (train_df[normalize_features] > q_low) &
        (train_df[normalize_features] < q_high)
    ]

    # 计算过滤后数据的统计量
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()

    # 将NaN值替换为0
    train_mean = train_mean.fillna(0.0)
    train_std = train_std.fillna(0.0)
    train_median = train_median.fillna(0.0)

    # 计算默认填充值
    default_fill = (train_median - train_mean) / (train_std + 1e-12)

    # 处理住院时长(LOS)信息
    los_info = None
    if "LOS" in train_df.columns:
        los_info = {
            "los_mean": train_mean["LOS"].item(),
            "los_std": train_std["LOS"].item(),
            "los_median": train_median["LOS"].item()
        }

        # 计算大住院时长和阈值(为covid-19基准设计)
        los_array = train_df.groupby(id_column)['LOS'].max().values
        los_p95 = np.percentile(los_array, 95)
        los_p5 = np.percentile(los_array, 5)
        filtered_los = los_array[(los_array >= los_p5) & (los_array <= los_p95)]
        los_info.update({
            "large_los": los_p95.item(),
            "threshold": filtered_los.mean().item() * 0.5
        })

    # 创建副本
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # 对数据集进行Z-score标准化
    for df in [train_df, val_df, test_df]:
        # 将数据类型转换为float
        df[normalize_features] = df[normalize_features].astype(float)
        # 标准化处理
        df.loc[:, normalize_features] = (df[normalize_features] - train_mean) / (train_std + 1e-12)
        # 过滤异常值
        df.loc[:, normalize_features] = df[normalize_features].map(filter_outlier)

    return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std