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
basic_records = ['RecordID', 'PatientID', 'RecordTime']
target_features = ['Outcome', 'LOS', 'Readmission']
note_features = ['Text']
demographic_features = ['Sex', 'Age']
labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response', 'Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
categorical_labtest_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
numerical_labtest_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
normalize_features = ['Age'] + numerical_labtest_features + ['LOS']

# Stratified split dataset into train, validation and test sets
# For ml/dl models: include Imputation & Normalization & Outlier Filtering steps
# For all settings, randomly select 200 patients for test set
# Then randomly select 10000 in the rest used for training and validation (7/8 training, 1/8 validation)

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic-iv-timeseries-note.parquet'))
df = df[basic_records + target_features + note_features + demographic_features + labtest_features]

# Group the dataframe by `RecordID`
grouped = df.groupby('RecordID')

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in patients])

# Randomly select 200 patients for the test set
train_val_patients, test_patients = train_test_split(patients, test_size=200, random_state=SEED, stratify=patients_outcome)
test_df = df[df['RecordID'].isin(test_patients)]

# For llm setting, export data on test set:
# Export the missing mask
test_missing_mask = export_missing_mask(test_df, demographic_features, labtest_features, id_column='RecordID')

# Export the record time
test_record_time = export_record_time(test_df, id_column='RecordID')

# Export the raw data
_, test_raw_x, _, _ = forward_fill_pipeline(test_df, None, demographic_features, labtest_features, target_features, [], id_column='RecordID')

# For ml/dl models, convert categorical features to one-hot encoding
one_hot = pd.get_dummies(df[categorical_labtest_features], columns=categorical_labtest_features, prefix_sep='->', dtype=float)
columns = df.columns.to_list()
column_start_idx = columns.index(categorical_labtest_features[0])
column_end_idx = columns.index(categorical_labtest_features[-1])
df = pd.concat([df.loc[:, columns[:column_start_idx]], one_hot, df.loc[:, columns[column_end_idx + 1:]]], axis=1)

# Update the categorical lab test features
ehr_categorical_labtest_features = one_hot.columns.to_list()
ehr_labtest_features = ehr_categorical_labtest_features + numerical_labtest_features
require_impute_features = ehr_labtest_features

# Group the dataframe by patient ID
grouped = df.groupby('RecordID')

# Randomly select 10000 patients for the train/val set
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_val_patients = train_test_split(train_val_patients, test_size=10000, random_state=SEED, stratify=train_val_patients_outcome)[1]

# Split the train/val set into train and val sets, 7/8 for train and 1/8 for val
train_val_patients_outcome = np.array([grouped.get_group(patient_id)['Outcome'].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=1/8, random_state=SEED, stratify=train_val_patients_outcome)

# Create train, val dataframes
train_df = df[df['RecordID'].isin(train_patients)]
val_df = df[df['RecordID'].isin(val_patients)]
test_df = df[df['RecordID'].isin(test_patients)]

# Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = normalize_dataframe(train_df, val_df, test_df, normalize_features, id_column="RecordID")

# Forward Imputation after grouped by RecordID
# Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
train_df, train_x, train_y, train_pid = forward_fill_pipeline(train_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
val_df, val_x, val_y, val_pid = forward_fill_pipeline(val_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")
test_df, test_x, test_y, test_pid = forward_fill_pipeline(test_df, default_fill, demographic_features, ehr_labtest_features, target_features, require_impute_features, id_column="RecordID")

# Export the note
train_note = export_note(train_df, id_column='RecordID')
val_note = export_note(val_df, id_column='RecordID')
test_note = export_note(test_df, id_column='RecordID')

# Create the directory to save the processed data
save_dir = os.path.join(processed_data_dir, 'split')
os.makedirs(save_dir, exist_ok=True)

# Convert the data to the required format
train_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, note_item, y_item in zip(train_pid, train_x, train_note, train_y)]
val_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, note_item, y_item in zip(val_pid, val_x, val_note, val_y)]
test_data = [{
    'id': id_item,
    'x_ts': x_item,
    'x_note': note_item,
    'x_llm_ts': x_llm_item,
    'record_time': record_time_item,
    'missing_mask': missing_mask_item,
    'y_mortality': [y[0] for y in y_item],
    'y_los': [y[1] for y in y_item],
    'y_readmission': [y[2] for y in y_item],
} for id_item, x_item, x_llm_item, note_item, record_time_item, missing_mask_item, y_item in zip(test_pid, test_x, test_raw_x, test_note, test_record_time, test_missing_mask, test_y)]

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
pd.to_pickle(labtest_features, os.path.join(save_dir, "labtest_features.pkl"))