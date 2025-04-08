# Import necessary libraries
import pandas as pd
import numpy as np

# Read the parquet file
ds = pd.read_parquet("my_datasets/mimic-iv/mimic-iv-timeseries_ehr-note.parquet")

# Display basic information about the dataset
print("\nDataset Info:")
print(ds.info())

# Display the first few rows
print("\nFirst few rows:")
print(ds.head())

# Display column names
print("\nColumns:")
print(ds.columns.tolist())

# Display basic statistics
print("\nBasic statistics:")
print(ds.describe())

# Display data types of columns
print("\nColumn data types:")
print(ds.dtypes)

# Calculate visits per patient
visits_per_patient = ds.groupby(['PatientID','AdmissionID']).size()

# Calculate statistics
avg_visits = visits_per_patient.mean()
median_visits = visits_per_patient.median()
max_visits = visits_per_patient.max()
min_visits = visits_per_patient.min()

print("\nVisits per patient statistics:")
print(f"Average visits per patient: {avg_visits:.2f}")
print(f"Median visits per patient: {median_visits:.2f}")
print(f"Maximum visits: {max_visits}")
print(f"Minimum visits: {min_visits}")

# Distribution of visit counts
visit_distribution = visits_per_patient.value_counts().sort_index()
print("\nDistribution of visits:")
print(visit_distribution)

# Calculate missing values for each column
missing_stats = {
    'Column': [],
    'Missing Count': [],
    'Missing Percentage': [],
    'Total Records': []
}

total_records = len(ds)

for column in ds.columns:
    missing_count = ds[column].isna().sum()
    missing_percentage = (missing_count / total_records) * 100

    missing_stats['Column'].append(column)
    missing_stats['Missing Count'].append(missing_count)
    missing_stats['Missing Percentage'].append(round(missing_percentage, 2))
    missing_stats['Total Records'].append(total_records)

# Create DataFrame with missing value statistics
missing_df = pd.DataFrame(missing_stats)

# Sort by missing percentage in descending order
missing_df = missing_df.sort_values('Missing Percentage', ascending=False)

# Display results
print("\nMissing Value Analysis:")
print(missing_df.to_string(index=False))
