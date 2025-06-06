SYSTEMPROMPT = {
    'tjh': 'You are an experienced doctor specializing in COVID-19 treatment, skilled in interpreting longitudinal patient data and predicting clinical outcomes.',
    'mimic-iv': 'You are an experienced critical care physician working in an Intensive Care Unit (ICU), skilled in interpreting complex longitudinal patient data and predicting clinical outcomes.',
}

TASK_DESCRIPTION = {
    'mortality': 'Your primary task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay.',
    'los': 'Your primary task is to analyze the medical data to predict the length of stay (LOS) in the hospital. The LOS is defined as the number of days from admission to discharge, including any days spent in the ICU.',
    'readmission': 'Your primary task is to analyze the medical data to predict the probability of readmission within 30 days post-discharge. Include cases where a patient passes away within 30 days from the discharge date as readmissions.',
}

RESPONSE_FORMAT = {
    'mortality': '''\
Please first perform a step-by-step analysis of the patient data, considering trends, abnormal values relative to reference ranges, and their clinical significance for survival. Then, provide a final assessment of the likelihood of not surviving the hospital stay.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number between 0 and 1 representing the predicted probability of mortality (higher value means higher likelihood of death).

Example Format: ```json { "think": "The patient presents with worsening X, stable Y, and improved Z. Factor A is a major risk indicator... Overall assessment suggests a high risk.", "answer": 0.85 }```''',

    'los': '''\
Please first perform a step-by-step analysis of the patient data, considering trends, abnormal values relative to reference ranges, and their clinical significance for length of stay. Then, provide a final assessment of the predicted length of stay in days.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number representing the predicted length of stay in days (higher value means longer stay).

Example Format: ```json { "think": "The patient shows stable vitals, but lab results indicate potential complications. Assuming no major changes, the predicted length of stay is...", "answer": 7 }```''',

    'readmission': '''\
Please first perform a step-by-step analysis of the patient data, considering trends, abnormal values relative to reference ranges, and their clinical significance for post-discharge stability and potential complications leading to readmission. Then, provide a final assessment of the likelihood of readmission within 30 days post-discharge.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number between 0 and 1 representing the predicted probability of readmission (higher value means higher likelihood of readmission).

Example Format: ```json { "think": "The patient's condition stabilized, but factors X and Y suggest potential post-discharge issues... Overall assessment indicates a moderate risk of readmission.", "answer": 0.40 }```''',
}

EXAMPLE = {
    'tjh': {
        'mortality': '''\
Input information of a patient:
The patient is a male, aged 52.0 years.
The patient had 5 visits that occurred at 2020-02-09, 2020-02-10, 2020-02-13, 2020-02-14, 2020-02-17.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI (Unit: ng/L. Reference range: less than 14.): [1.9, 1.9, 1.9, 1.9, 1.9]
- hemoglobin (Unit: g/L. Reference range: 140 - 180 for men, 120 - 160 for women.): [139.0, 139.0, 142.0, 142.0, 142.0]
- Serum chloride (Unit: mmol/L. Reference range: 96 - 106.): [103.7, 103.7, 104.2, 104.2, 104.2]
...... (other features omitted for brevity)

Response:
```json { "think": "The patient is a 52-year-old male. Key labs like Troponin I are consistently normal and low. Hemoglobin is borderline low for a male but stable. Serum chloride is within normal limits and stable. Assuming other unlisted vital signs and labs are also stable or within normal limits, the overall picture suggests a relatively stable condition without indicators of severe organ damage or rapid decline commonly associated with high mortality risk in this context. The risk appears low.", "answer": 0.25 } ```
''',
    'los': '''\
Input information of a patient:
The patient is a male, aged 52.0 years.
The patient had 5 visits that occurred at 2020-02-09, 2020-02-10, 2020-02-13, 2020-02-14, 2020-02-17.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI (Unit: ng/L. Reference range: less than 14.): [1.9, 1.9, 1.9, 1.9, 1.9]
- hemoglobin (Unit: g/L. Reference range: 140 - 180 for men, 120 - 160 for women.): [139.0, 139.0, 142.0, 142.0, 142.0]
- Serum chloride (Unit: mmol/L. Reference range: 96 - 106.): [103.7, 103.7, 104.2, 104.2, 104.2]
...... (other features omitted for brevity)

Response:
```json {"think": "The patient is a 52-year-old male. The provided data spans 5 visits over an 8-day period (Feb 9 to Feb 17), indicating the stay is at least this long. Key available labs show stability: Hypersensitive cardiac troponin I is consistently low and normal, suggesting no acute cardiac injury. A prediction around 10-14 days seems reasonable given the stability but also the duration already observed.", "answer": 12.0 } ```
'''
    },
    'mimic-iv': {
        'mortality': '''\
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate (Unit: /. Reference range: /.): ["unknown", "unknown", "unknown", "unknown"]
- Glascow coma scale eye opening (Unit: /. Reference range: /.): ["Spontaneously", "Spontaneously", "Spontaneously", "Spontaneously"]
- Glascow coma scale motor response (Unit: /. Reference range: /.): ["Obeys Commands", "Obeys Commands", "Obeys Commands", "Obeys Commands"]
...... (other features omitted for brevity)

Response:
```json { "think": "Patient is 52 years old. GCS components indicate full alertness and responsiveness (spontaneous eye opening, obeys commands) consistently across the recorded time points. While capillary refill is unknown, the neurological status appears stable and good. Assuming other vital signs and labs (not shown) are not critically deranged, the current data suggests a lower risk of mortality.", "answer": 0.3 } ```
''',
        'readmission': '''\
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate (Unit: /. Reference range: /.): ["unknown", "unknown", "unknown", "unknown"]
- Glascow coma scale eye opening (Unit: /. Reference range: /.): ["Spontaneously", "Spontaneously", "Spontaneously", "Spontaneously"]
- Glascow coma scale motor response (Unit: /. Reference range: /.): ["Obeys Commands", "Obeys Commands", "Obeys Commands", "Obeys Commands"]
...... (other features omitted for brevity)

Response:
```json { "think": "The patient (52F) shows stable and normal neurological status (GCS) throughout the observed period. Assuming clinical stability in other domains leading to discharge, the risk of readmission based solely on this neurological data appears relatively low. However, without information on the primary diagnosis, comorbidities, or discharge status, a definitive low risk cannot be assumed. Assigning a moderate-low probability.", "answer": 0.3 } ```
'''
    },
}

USERPROMPT = """\
I will provide you with longitudinal medical information for a patient. The data covers {LENGTH} visits that occurred at {RECORD_TIME_LIST}.
Each clinical feature is presented as a list of values, corresponding to these visits. Missing values are represented as `NaN` for numerical values and "unknown" for categorical values. Note that units and reference ranges are provided alongside relevant features.

Patient Background:
- Sex: {SEX}
- Age: {AGE} years

Your Task:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Handling Uncertainty:
In situations where the provided data is clearly insufficient or too ambiguous to make a reasonable prediction, respond with the exact phrase: `I do not know`.

{EXAMPLE}

Now, please analyze and predict for the following patient:

Clinical Features Over Time:
{DETAIL}"""

UNIT = {
    'tjh': 'src/structured_ehr/prompts/tjh/unit.json',
    'mimic-iv': 'src/structured_ehr/prompts/mimic-iv/unit.json',
}

REFERENCE_RANGE = {
    'tjh': 'src/structured_ehr/prompts/tjh/range.json',
    'mimic-iv': 'src/structured_ehr/prompts/mimic-iv/range.json',
}