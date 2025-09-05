import os
import random
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

processed_data_dir = os.path.join("./my_datasets/mimic-iii", 'processed')
os.makedirs(processed_data_dir, exist_ok=True)
SEED = 42

# Record feature names
basic_records = ["RecordID", "PatientID"]
target_features = ["InHospitalOutcome"]
note_features = ["Text"]

# Stratified split dataset into train, validation and test sets
# For all settings, randomly select 200 patients for test set
# Then randomly select 10000 in the rest used for training and validation (7/8 training, 1/8 validation)

# Read the dataset
df = pd.read_parquet(os.path.join(processed_data_dir, 'mimic_iii_note_label.parquet'))
df = df[basic_records + target_features + note_features]

# Ensure the data is sorted by RecordID
df = df.sort_values(by=["RecordID"]).reset_index(drop=True)

# Group the dataframe by `RecordID`
grouped = df.groupby("RecordID")

# Get the patient IDs and outcomes
patients = np.array(list(grouped.groups.keys()))
patients_outcome = np.array([grouped.get_group(patient_id)["InHospitalOutcome"].iloc[0] for patient_id in patients])

# Randomly select 200 patients for the test set
train_val_patients, test_patients = train_test_split(patients, test_size=200, random_state=SEED, stratify=patients_outcome)

# Randomly select 10000 patients for the train/val set
train_val_patients_outcome = np.array([grouped.get_group(patient_id)["InHospitalOutcome"].iloc[0] for patient_id in train_val_patients])
train_val_patients = train_test_split(train_val_patients, test_size=10000, random_state=SEED, stratify=train_val_patients_outcome)[1]

# Split the train/val set into train and val sets, 7/8 for train and 1/8 for val
train_val_patients_outcome = np.array([grouped.get_group(patient_id)["InHospitalOutcome"].iloc[0] for patient_id in train_val_patients])
train_patients, val_patients = train_test_split(train_val_patients, test_size=1/8, random_state=SEED, stratify=train_val_patients_outcome)

# Print the sizes of the datasets
print("Train patients size:", len(train_patients))
print("Validation patients size:", len(val_patients))
print("Test patients size:", len(test_patients))

# Assert there is no data leakage
assert len(set(train_patients) & set(val_patients)) == 0, "Data leakage between train and val sets"
assert len(set(train_patients) & set(test_patients)) == 0, "Data leakage between train and test sets"
assert len(set(val_patients) & set(test_patients)) == 0, "Data leakage between val and test sets"

# Create train, val dataframes
train_df = df[df["RecordID"].isin(train_patients)]
val_df = df[df["RecordID"].isin(val_patients)]
test_df = df[df["RecordID"].isin(test_patients)]

# Create the directory to save the processed data
save_dir = os.path.join(processed_data_dir, "fold_llm")
os.makedirs(save_dir, exist_ok=True)

# Convert the data to the required format
train_data = [{
    "id": id_item,
    "x_note": note_item,
    "y_mortality": y_item.item(),
} for id_item, note_item, y_item in zip(train_patients.tolist(), train_df["Text"].values, train_df["InHospitalOutcome"].values)]
val_data = [{
    "id": id_item,
    "x_note": note_item,
    "y_mortality": y_item.item(),
} for id_item, note_item, y_item in zip(val_patients.tolist(), val_df["Text"].values, val_df["InHospitalOutcome"].values)]
test_data = [{
    "id": id_item,
    "x_note": note_item,
    "y_mortality": y_item.item(),
} for id_item, note_item, y_item in zip(test_patients.tolist(), test_df["Text"].values, test_df["InHospitalOutcome"].values)]

# Save the data to pickle files
pd.to_pickle(train_data, os.path.join(save_dir, "train_data.pkl"))
pd.to_pickle(val_data, os.path.join(save_dir, "val_data.pkl"))
pd.to_pickle(test_data, os.path.join(save_dir, "test_data.pkl"))

# Print the sizes of the datasets
print("Train data size:", len(train_data))
print("Validation data size:", len(val_data))
print("Test data size:", len(test_data))