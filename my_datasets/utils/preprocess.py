from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd


def calculate_data_existing_length(data: np.ndarray) -> int:
    """
    Calculate the number of non-missing (non-NaN) values in the data.

    Args:
        data: Input data array

    Returns:
        Count of non-NaN values
    """
    return sum(1 for item in data if not pd.isna(item))


def fill_missing_value(data: np.ndarray, to_fill_value: float = 0) -> np.ndarray:
    """
    Fill missing values in time series data sorted in ascending order by time.
    Uses specified fill value for leading missing values at the beginning of the sequence,
    and forward fill strategy for subsequent missing values.

    Args:
        data: Data array sorted in ascending order by time
        to_fill_value: Default value for filling leading missing values

    Returns:
        Filled data array
    """
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)

    # Return original data if no missing values
    if data_len == data_exist_len:
        return data

    # Fill all with specified value if all values are missing
    if data_exist_len == 0:
        data[:] = to_fill_value
        return data

    # Handle leading missing values at sequence start
    if pd.isna(data[0]):
        # Find position of first non-NaN value
        not_na_pos = next((i for i, val in enumerate(data) if not pd.isna(val)), 0)
        # Fill elements before first non-NaN value
        data[:not_na_pos] = to_fill_value

    # Apply forward fill for remaining missing values
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
    Forward filling pipeline for patient time series data. Sorts each patient's time series,
    fills missing values, and extracts features/target variables.

    Args:
        df: DataFrame containing patient data
        default_fill: DataFrame with default fill values
        demographic_features: List of demographic features
        labtest_features: List of lab test features
        target_features: List of target variable features
        require_impute_features: Features requiring missing value imputation

    Returns:
        Tuple containing:
            - Filled DataFrame
            - List of features for all patients
            - List of target variables for all patients
            - List of all patient IDs
    """
    df_copy = df.copy()
    grouped = df_copy.groupby(id_column)
    all_x, all_y, all_pid = [], [], []

    for patient_id, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)

        # Handle missing values for features requiring imputation
        for feature in require_impute_features:
            # Determine fill value, default -1 for categorical features
            to_fill_value = default_fill.get(feature, -1)
            # Use patient median for missing value imputation
            fill_missing_value(sorted_group[feature].values, to_fill_value)

        # Extract features and target variables
        patient_x = []
        patient_y = []

        for _, row in sorted_group.iterrows():
            # Extract target variables
            target_values = [row[f] for f in target_features]
            patient_y.append(target_values)

            # Extract features
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
    Export missing value mask indicating missing values for each feature.

    Args:
        df: DataFrame containing patient data
        demographic_features: List of demographic features
        labtest_features: List of lab test features

    Returns:
        List of missing value masks
    """
    df_copy = df.copy()
    # Ensure the data is sorted by ID and RecordTime
    df_copy = df_copy.sort_values(by=[id_column, "RecordTime"]).reset_index(drop=True)

    grouped = df.groupby(id_column)
    missing_mask = []

    for _, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)

        # Generate mask for each patient
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
    Export record time sequences grouped by patient ID.

    Args:
        df: DataFrame containing patient data

    Returns:
        List of record time sequences
    """
    df_copy = df.copy()
    # Ensure the data is sorted by ID and RecordTime
    df_copy = df_copy.sort_values(by=[id_column, "RecordTime"]).reset_index(drop=True)
    # Convert and format RecordTime
    df_copy["RecordTime"] = pd.to_datetime(df_copy["RecordTime"]).dt.strftime("%Y-%m-%d")

    grouped = df_copy.groupby(id_column)
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
    Export clinical notes grouped by patient ID.

    Args:
        df: DataFrame containing patient data

    Returns:
        List of clinical notes
    """
    df_copy = df.copy()
    # Ensure the data is sorted by ID and RecordTime
    df_copy = df_copy.sort_values(by=[id_column, "RecordTime"]).reset_index(drop=True)

    grouped = df.groupby(id_column)
    notes = grouped.first()[note_column].tolist()

    return notes


def filter_outlier(element: float) -> float:
    """
    Filter outliers by replacing values with absolute value >10000 with 0.

    Args:
        element: Input value

    Returns:
        Filtered value
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
    Normalize features across train/val/test sets and compute related statistics.

    Args:
        train_df: Training set DataFrame
        val_df: Validation set DataFrame
        test_df: Test set DataFrame
        normalize_features: List of features to normalize

    Returns:
        Tuple containing:
            - Normalized training set
            - Normalized validation set
            - Normalized test set
            - Default fill values DataFrame
            - LOS-related info dict (if exists)
            - Training set mean
            - Training set std
    """
    # Calculate quantiles to filter outliers
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)
    filtered_df = train_df[
        (train_df[normalize_features] > q_low) &
        (train_df[normalize_features] < q_high)
    ]

    # Compute statistics from filtered data
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()

    # Replace NaNs with 0
    train_mean = train_mean.fillna(0.0)
    train_std = train_std.fillna(0.0)
    train_median = train_median.fillna(0.0)

    # Compute default fill values
    default_fill = (train_median - train_mean) / (train_std + 1e-12)

    # Handle Length of Stay (LOS) information
    los_info = None
    if "LOS" in train_df.columns:
        los_info = {
            "los_mean": train_mean["LOS"].item(),
            "los_std": train_std["LOS"].item(),
            "los_median": train_median["LOS"].item()
        }

        # Calculate large LOS and threshold (for covid-19 benchmark design)
        los_array = train_df.groupby(id_column)['LOS'].max().values
        los_p95 = np.percentile(los_array, 95)
        los_p5 = np.percentile(los_array, 5)
        filtered_los = los_array[(los_array >= los_p5) & (los_array <= los_p95)]
        los_info.update({
            "large_los": los_p95.item(),
            "threshold": filtered_los.mean().item() * 0.5
        })

    # Create copies
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Apply Z-score normalization
    for df in [train_df, val_df, test_df]:
        # Convert data types
        df[normalize_features] = df[normalize_features].astype(float)
        # Normalization
        df.loc[:, normalize_features] = (df[normalize_features] - train_mean) / (train_std + 1e-12)
        # Filter outliers
        df.loc[:, normalize_features] = df[normalize_features].map(filter_outlier)

    return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std