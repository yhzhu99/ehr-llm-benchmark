import os
import random
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
# For ml/dl models: include Imputation & Normalization & Outlier Filtering steps
# For all settings, randomly select 200 patients for test set
# The rest are used for training and validation (7/8 training, 1/8 validation)

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'tjh_dataset_formatted.parquet'))

# Ensure the data is sorted by PatientID and RecordTime
df = df.sort_values(by=['PatientID', 'RecordTime']).reset_index(drop=True)

# Group the dataframe by `PatientID`
grouped = df.groupby('PatientID')

# Get the patient IDs
patients = np.array(list(grouped.groups.keys()))

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Randomly select 200 patients for the test set
train_val_patients, test_patients = train_test_split(patients, test_size=200, random_state=SEED)

# Get the remaining patients for the train/val set
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=1/8, random_state=SEED, stratify=train_val_patients_outcome)

# Create train, val, test, dataframes
train_df = df[df['PatientID'].isin(train_patients)]
val_df = df[df['PatientID'].isin(val_patients)]
test_df = df[df['PatientID'].isin(test_patients)]

# For llm setting, export data on test set:
# Export the missing mask
train_missing_mask = export_missing_mask(train_df, demographic_features, labtest_features)
val_missing_mask = export_missing_mask(val_df, demographic_features, labtest_features)
test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features)

# Export the record time
train_record_time = export_record_time(train_df)
val_record_time = export_record_time(val_df)
test_record_time = export_record_time(test_df)

# Export the raw data
_, train_raw_x, _, _ = forward_fill_pipeline(train_df, None, demographic_features, labtest_features, target_features, [])
_, val_raw_x, _, _ = forward_fill_pipeline(val_df, None, demographic_features, labtest_features, target_features, [])
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [])

# For dl setting, export data on train/val/test set:
# Normalize the train, val, test data
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features)

# Forward Imputation after grouped by PatientID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, labtest_features, target_features, require_impute_features)

# Convert the data to the required format
train_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
} for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(train_pid, train_x, train_raw_x, train_record_time, train_missing_mask, train_y)]
val_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
} for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(val_pid, val_x, val_raw_x, val_record_time, val_missing_mask, val_y)]
test_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
} for id_item, x_item, x_llm_item, record_time_item, missing_mask_item, y_item in zip(test_pid, test_x, test_raw_x, test_record_time, test_missing_mask, test_y)]

# Create the directory to save the processed data
save_dir = os.path.join(processed_data_dir, 'split')
os.makedirs(save_dir, exist_ok=True)

# Save the data to pickle files
pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
pd.to_pickle(test_data, os.path.join(save_dir, "test_data.pkl"))

# Print the sizes of the datasets
print("Train data size:", len(train_data))
print("Validation data size:", len(val_data))
print("Test data size:", len(test_data))

# Export LOS statistics (calculated from the train set)
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))

# Export the labtest feature names
def process_labtest_feature_name(name: str):
    # Remove special characters and extra spaces
    name = name.strip().replace('-', '')
    return name
labtest_features = [process_labtest_feature_name(name) for name in labtest_features]
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))

# Extract 10 shots (5 pos nad 5 neg) from train set and valid set
train_pos_data = [item for item in train_data if item['y_mortality'][0] == 1]
train_neg_data = [item for item in train_data if item['y_mortality'][0] == 0]
val_pos_data = [item for item in val_data if item['y_mortality'][0] == 1]
val_neg_data = [item for item in val_data if item['y_mortality'][0] == 0]
train_pos_data = random.sample(train_pos_data, min(5, len(train_pos_data)))
train_neg_data = random.sample(train_neg_data, min(5, len(train_neg_data)))
val_pos_data = random.sample(val_pos_data, min(5, len(val_pos_data)))
val_neg_data = random.sample(val_neg_data, min(5, len(val_neg_data)))
train_shot_data = train_pos_data + train_neg_data
val_shot_data = val_pos_data + val_neg_data

save_dir = os.path.join(processed_data_dir, '10_shot')
os.makedirs(save_dir, exist_ok=True)

# Save the data to pickle files
pd.to_pickle(train_shot_data, os.path.join(save_dir, "train_data.pkl"))
pd.to_pickle(val_shot_data, os.path.join(save_dir, "val_data.pkl"))
pd.to_pickle(test_data, os.path.join(save_dir, "test_data.pkl"))
pd.to_pickle(los_info, os.path.join(save_dir, "los_info.pkl"))