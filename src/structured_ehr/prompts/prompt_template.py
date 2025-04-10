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

Example Format: `{ "think": "The patient presents with worsening X, stable Y, and improved Z. Factor A is a major risk indicator... Overall assessment suggests a high risk.", "answer": 0.85 }`''',

    'los': '''\
Please first perform a step-by-step analysis of the patient data, considering trends, abnormal values relative to reference ranges, and their clinical significance for length of stay. Then, provide a final assessment of the predicted length of stay in days.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number representing the predicted length of stay in days (higher value means longer stay).

Example Format: `{ "think": "The patient shows stable vitals, but lab results indicate potential complications. Assuming no major changes, the predicted length of stay is...", "answer": 7.5 }`''',

    'readmission': '''\
Please first perform a step-by-step analysis of the patient data, considering trends, abnormal values relative to reference ranges, and their clinical significance for post-discharge stability and potential complications leading to readmission. Then, provide a final assessment of the likelihood of readmission within 30 days post-discharge.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number between 0 and 1 representing the predicted probability of readmission (higher value means higher likelihood of readmission).

Example Format: `{ "think": "The patient's condition stabilized, but factors X and Y suggest potential post-discharge issues... Overall assessment indicates a moderate risk of readmission.", "answer": 0.40 }`''',
}

EXAMPLE = {
    'tjh': {
        'mortality': [
            '''
Input information of a patient:
The patient is a male, aged 52.0 years.
The patient had 5 visits that occurred at 2020-02-09, 2020-02-10, 2020-02-13, 2020-02-14, 2020-02-17.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI (Unit: ng/L. Reference range: less than 14.): [1.9, 1.9, 1.9, 1.9, 1.9]
- hemoglobin (Unit: g/L. Reference range: 140 - 180 for men, 120 - 160 for women.): [139.0, 139.0, 142.0, 142.0, 142.0]
- Serum chloride (Unit: mmol/L. Reference range: 96 - 106.): [103.7, 103.7, 104.2, 104.2, 104.2]
...... (other features omitted for brevity)

{ "think": "The patient is a 52-year-old male. Key labs like Troponin I are consistently normal and low. Hemoglobin is borderline low for a male but stable. Serum chloride is within normal limits and stable. Assuming other unlisted vital signs and labs are also stable or within normal limits, the overall picture suggests a relatively stable condition without indicators of severe organ damage or rapid decline commonly associated with high mortality risk in this context. The risk appears low.", "answer": 0.25 }
''',
            '''
Input information of a patient:
The patient is a female, aged 71.0 years.
The patient had 5 visits that occurred at 2020-02-01, 2020-02-02, 2020-02-09, 2020-02-10, 2020-02-11.
Details of the features for each visit are as follows:
- Hypersensitive cardiac troponinI (Unit: ng/L. Reference range: less than 14.): [5691.05, 11970.22, 9029.88, 6371.5, 3638.55]
- hemoglobin (Unit: g/L. Reference range: 140 - 180 for men, 120 - 160 for women.): [105.68, 132.84, 54.19, 136.33, 123.69]
- Serum chloride (Unit: mmol/L. Reference range: 96 - 106.): [89.18, 101.54, 90.35, 103.99, 102.06]
...... (other features omitted for brevity)

RESPONSE:
{ "think": "The patient is an elderly female (71 years) with significantly elevated Troponin I levels across all visits, peaking extremely high, indicating severe myocardial injury. Hemoglobin shows significant fluctuations, including a very low value (54.19 g/L), suggesting potential bleeding or severe anemia requiring intervention. Serum chloride levels also fluctuate, dipping below the normal range. Advanced age combined with evidence of severe cardiac damage and hemodynamic instability (indicated by Hgb fluctuations) points towards a very high risk of mortality.", "answer": 0.85 }
''',
            # ... Add more examples in the correct JSON format
        ],
    },
    'mimic-iv': {
        'mortality': [
             '''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: ["unknown", "unknown", "unknown", "unknown"]
- Glascow coma scale eye opening: ["Spontaneously", "Spontaneously", "Spontaneously", "Spontaneously"]
- Glascow coma scale motor response: ["Obeys Commands", "Obeys Commands", "Obeys Commands", "Obeys Commands"]
...... (other features omitted for brevity)

{ "think": "Patient is 52 years old. GCS components indicate full alertness and responsiveness (spontaneous eye opening, obeys commands) consistently across the recorded time points. While capillary refill is unknown, the neurological status appears stable and good. Assuming other vital signs and labs (not shown) are not critically deranged, the current data suggests a lower risk of mortality.", "answer": 0.3 }
''',
             '''
Input information of a patient:
The patient is a male, aged 49 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: ["unknown", "unknown", "unknown", "unknown"]
- Glascow coma scale eye opening: ["To speech", "To speech", "To speech", "Spontaneously"]
- Glascow coma scale motor response: ["Abnorm extensn", "Obeys Commands", "No Response", "Localizes Pain"]
...... (other features omitted for brevity)

{ "think": "Patient is 49 years old. GCS shows significant fluctuation and severe impairment. Eye opening requires speech initially, motor response includes abnormal extension and later no response, indicating severe neurological compromise (low GCS score). Although there's some later improvement (spontaneous eye opening, localizes pain), the period of severe deficit, particularly 'No Response' motor, is a very poor prognostic sign in the ICU. This neurological instability indicates a high risk of mortality.", "answer": 0.9 }
''',
        ],
        'readmission': [
            '''
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Capillary refill rate: ["unknown", "unknown", "unknown", "unknown"]
- Glascow coma scale eye opening: ["Spontaneously", "Spontaneously", "Spontaneously", "Spontaneously"]
- Glascow coma scale motor response: ["Obeys Commands", "Obeys Commands", "Obeys Commands", "Obeys Commands"]
...... (other features omitted for brevity)

{ "think": "The patient (52F) shows stable and normal neurological status (GCS) throughout the observed period. Assuming clinical stability in other domains leading to discharge, the risk of readmission based solely on this neurological data appears relatively low. However, without information on the primary diagnosis, comorbidities, or discharge status, a definitive low risk cannot be assumed. Assigning a moderate-low probability.", "answer": 0.3 }
'''
        ],
    },
}

USERPROMPT = """\
I will provide you with longitudinal medical information for a patient. The data covers {LENGTH} visits/time points that occurred at {RECORD_TIME_LIST}.
Each clinical feature is presented as a list of values, corresponding to these time points. Missing values are represented as `NaN` for numerical values and "unknown" for categorical values. Note that units and reference ranges are provided alongside relevant features.

Patient Background:
- Sex: {SEX}
- Age: {AGE} years

Your Task:
{TASK_DESCRIPTION}

Instructions & Output Format:
{RESPONSE_FORMAT}

Handling Uncertainty:
In situations where the provided data is clearly insufficient or too ambiguous to make a reasonable prediction, respond with the exact phrase: `I do not know`

Examples:
{EXAMPLE}

Now, please analyze and predict for the following patient:

Clinical Features Over Time:
{DETAIL}"""

UNIT = {
    'tjh': 'prompts/tjh/unit.json',
    'mimic-iv': 'prompts/mimic-iv/unit.json',
}

REFERENCE_RANGE = {
    'tjh': 'prompts/tjh/range.json',
    'mimic-iv': 'prompts/mimic-iv/range.json',
}