import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time, export_note

data_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

# Record feature names
basic_records = ['PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
demographic_features = ['Sex', 'Age']
labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
require_impute_features = labtest_features
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# For LLM, 200 samples are randomly selected for test set, and the rest are used for training and validation

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic4_discharge_note_ehr.parquet'))

# Group the dataframe by patient ID
grouped = df.groupby('PatientID')

# Get the patient IDs
patients = np.array(list(grouped.groups.keys()))

# Ramdomly select 200 patients for the test set
test_patients = np.random.choice(patients, size=200, replace=False)

# Get the remaining patients for the train/val set
train_val_patients = np.setdiff1d(patients, test_patients)
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=10/80, random_state=SEED, stratify=train_val_patients_outcome)

# Create train, val, test, dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
save_dir = os.path.join(processed_data_dir, 'fold_llm') # forward fill
os.makedirs(save_dir, exist_ok=True)

# Export the missing mask and record time
for split, df_split in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    # Export the record time
    record_time = export_record_time(df_split)
    pd.to_pickle(record_time, os.path.join(save_dir, f"{split}_record_time.pkl"))

    # Export the missing mask
    missing_mask = export_missing_mask(df_split, demographic_features, labtest_features)
    pd.to_pickle(missing_mask, os.path.join(save_dir, f"{split}_missing_mask.pkl"))

    # Export the note
    note = export_note(df_split)
    pd.to_pickle(note, os.path.join(save_dir, f"{split}_note.pkl"))

# Export the raw data
_, train_raw_x, _, _ = forward_fill_pipeline(train_df, None, demographic_features, labtest_features, target_features, [])
_, val_raw_x, _, _ = forward_fill_pipeline(val_df, None, demographic_features, labtest_features, target_features, [])
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [])

train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_raw_x, os.path.join(save_dir, "train_raw_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_raw_x, os.path.join(save_dir, "val_raw_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_raw_x, os.path.join(save_dir, "test_raw_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl")) # All features

# Stratified split dataset into `Training`, `Validation` and `Test` sets
# Stratified dataset according to `Outcome` column
# For ml/dl models, 70% Training, 10% Validation, 20% Test

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, "mimic4_discharge_note_ehr.parquet"))

# Change the order of columns
df = df[basic_records + target_features + demographic_features + categorical_labtest_features + numerical_labtest_features]

# For ml/dl models, convert categorical features to one-hot encoding
one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
columns = df.columns.to_list()
column_start_idx = columns.index(categorical_labtest_features[0])
column_end_idx = columns.index(categorical_labtest_features[-1])
df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)

# Update the categorical lab test features
categorical_labtest_features = one_hot.columns.to_list()
labtest_features = categorical_labtest_features + numerical_labtest_features
require_impute_features = labtest_features

# Group the dataframe by patient ID
grouped = df.groupby('PatientID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Get the train_val/test patient IDs
train_val_patients, test_patients = train_test_split(patients, test_size=20/100, random_state=SEED, stratify=patients_outcome)

# Get the train/val patient IDs
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=10/80, random_state=SEED, stratify=train_val_patients_outcome)

# Create train, val, test, [traincal, calib] dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
save_dir = os.path.join(processed_data_dir, 'fold_dl') # forward fill
os.makedirs(save_dir, exist_ok=True)

# Save the train, val, and test dataframes for the current fold to csv files
train_df.to_csv(os.path.join(save_dir, "train_raw.csv"), index=False)
val_df.to_csv(os.path.join(save_dir, "val_raw.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "test_raw.csv"), index=False)

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Save the zscored dataframes to csv files
train_df.to_csv(os.path.join(save_dir, "train_after_zscore.csv"), index=False)
val_df.to_csv(os.path.join(save_dir, "val_after_zscore.csv"), index=False)
test_df.to_csv(os.path.join(save_dir, "test_after_zscore.csv"), index=False)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Save the imputed dataset to pickle file
pd.to_pickle(train_x, os.path.join(save_dir, "train_x.pkl"))
pd.to_pickle(train_y, os.path.join(save_dir, "train_y.pkl"))
pd.to_pickle(train_pid, os.path.join(save_dir, "train_pid.pkl"))
pd.to_pickle(val_x, os.path.join(save_dir, "val_x.pkl"))
pd.to_pickle(val_y, os.path.join(save_dir, "val_y.pkl"))
pd.to_pickle(val_pid, os.path.join(save_dir, "val_pid.pkl"))
pd.to_pickle(test_x, os.path.join(save_dir, "test_x.pkl"))
pd.to_pickle(test_y, os.path.join(save_dir, "test_y.pkl"))
pd.to_pickle(test_pid, os.path.join(save_dir, "test_pid.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl")) # LOS statistics (calculated from the train set)
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl")) # All features