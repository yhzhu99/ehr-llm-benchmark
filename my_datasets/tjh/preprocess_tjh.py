import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocess import forward_fill_pipeline, normalize_dataframe, export_missing_mask, export_record_time


data_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

df = pd.read_excel(os.path.join(raw_data_dir, 'time_series_375_prerpocess_en.xlsx'))

# Rename columns
df = df.rename(columns={"PATIENT_ID": "PatientID", "outcome": "Outcome", "gender": "Sex", "age": "Age", "RE_DATE": "RecordTime", "Admission time": "AdmissionTime", "Discharge time": "DischargeTime"})

# Fill PatientID column
df['PatientID'] = df['PatientID'].ffill()

# Format data values
# gender transformation: 1--male, 0--female
df['Sex'] = df['Sex'].replace(2, 0)

# only reserve y-m-d precision for `RE_DATE` and `Discharge time` columns
df['RecordTime'] = df['RecordTime'].dt.strftime('%Y-%m-%d')
df['DischargeTime'] = df['DischargeTime'].dt.strftime('%Y-%m-%d')
df['AdmissionTime'] = df['AdmissionTime'].dt.strftime('%Y-%m-%d')

# Exclude patients with missing labels
df = df.dropna(subset = ['PatientID', 'RecordTime', 'DischargeTime'], how='any')

# Calculate the Length-of-Stay (LOS) label
df['LOS'] = (pd.to_datetime(df['DischargeTime']) - pd.to_datetime(df['RecordTime'])).dt.days

# Notice: Set negative LOS values to 0
df['LOS'] = df['LOS'].apply(lambda x: 0 if x < 0 else x)

# Drop columns whose values are all the same or all NaN
df = df.drop(columns=['2019-nCoV nucleic acid detection'])

# Record feature names
basic_records = ['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime']
target_features = ['Outcome', 'LOS']
demographic_features = ['Sex', 'Age']
labtest_features = ['Hypersensitive cardiac troponinI', 'hemoglobin', 'Serum chloride', 'Prothrombin time', 'procalcitonin', 'eosinophils(%)', 'Interleukin 2 receptor', 'Alkaline phosphatase', 'albumin', 'basophil(%)', 'Interleukin 10', 'Total bilirubin', 'Platelet count', 'monocytes(%)', 'antithrombin', 'Interleukin 8', 'indirect bilirubin', 'Red blood cell distribution width ', 'neutrophils(%)', 'total protein', 'Quantification of Treponema pallidum antibodies', 'Prothrombin activity', 'HBsAg', 'mean corpuscular volume', 'hematocrit', 'White blood cell count', 'Tumor necrosis factorα', 'mean corpuscular hemoglobin concentration', 'fibrinogen', 'Interleukin 1β', 'Urea', 'lymphocyte count', 'PH value', 'Red blood cell count', 'Eosinophil count', 'Corrected calcium', 'Serum potassium', 'glucose', 'neutrophils count', 'Direct bilirubin', 'Mean platelet volume', 'ferritin', 'RBC distribution width SD', 'Thrombin time', '(%)lymphocyte', 'HCV antibody quantification', 'D-D dimer', 'Total cholesterol', 'aspartate aminotransferase', 'Uric acid', 'HCO3-', 'calcium', 'Amino-terminal brain natriuretic peptide precursor(NT-proBNP)', 'Lactate dehydrogenase', 'platelet large cell ratio ', 'Interleukin 6', 'Fibrin degradation products', 'monocytes count', 'PLT distribution width', 'globulin', 'γ-glutamyl transpeptidase', 'International standard ratio', 'basophil count(#)', 'mean corpuscular hemoglobin ', 'Activation of partial thromboplastin time', 'High sensitivity C-reactive protein', 'HIV antibody quantification', 'serum sodium', 'thrombocytocrit', 'ESR', 'glutamic-pyruvic transaminase', 'eGFR', 'creatinine']
require_impute_features = labtest_features
normalize_features = ['Age'] + labtest_features + ['LOS']

# Set negative values to NaN
df[df[demographic_features + labtest_features] < 0] = np.nan

# Merge by date
df = df.groupby(['PatientID', 'RecordTime', 'AdmissionTime', 'DischargeTime'], dropna=True, as_index = False).mean()

# Change the order of columns
df = df[basic_records + target_features + demographic_features + labtest_features]

# Export data to files
df.to_parquet(os.path.join(processed_data_dir, 'tjh_dataset_formatted.parquet'), index=False)

# Stratified split dataset into train, validation and test sets
# Also include (Imputation & Normalization & Outlier Filtering) steps
# The train, validation and test sets are saved in the `./processed` folder
# For ml/dl models, use 7:1:2 splitting strategy, 70% training, 10% validation, 20% testing

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'tjh_dataset_formatted.parquet'))

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

# Create train, val, test dataframes for the current fold
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]
save_dir = os.path.join(processed_data_dir, 'fold_dl')
os.makedirs(save_dir, exist_ok=True)

# Save the train, val, and test dataframes for the current fold to parquet files
train_df.to_parquet(os.path.join(save_dir, "train_raw.parquet"), index=False)
val_df.to_parquet(os.path.join(save_dir, "val_raw.parquet"), index=False)
test_df.to_parquet(os.path.join(save_dir, "test_raw.parquet"), index=False)

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Save the zscored dataframes to parquet files
train_df.to_parquet(os.path.join(save_dir, "train_after_zscore.parquet"), index=False)
val_df.to_parquet(os.path.join(save_dir, "val_after_zscore.parquet"), index=False)
test_df.to_parquet(os.path.join(save_dir, "test_after_zscore.parquet"), index=False)

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

# For LLM, 200 samples are randomly selected for test set, and the rest are used for training and validation

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'tjh_dataset_formatted.parquet'))

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


