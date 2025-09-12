SYSTEMPROMPT = {
    'mimic-iv': 'You are an experienced critical care physician working in an Intensive Care Unit (ICU), skilled in interpreting complex longitudinal patient data, including time-series vitals and labs, as well as unstructured clinical notes to predict clinical outcomes.',
}

TASK_DESCRIPTION = {
    'mortality': 'Your primary task is to assess the provided medical data, including both the time-series clinical features and the unstructured clinical note, to determine the likelihood of the patient not surviving their hospital stay.',
    'readmission': 'Your primary task is to analyze the provided medical data, including both the time-series clinical features and the unstructured clinical note, to predict the probability of readmission within 30 days post-discharge. Include cases where a patient passes away within 30 days from the discharge date as readmissions.',
}

RESPONSE_FORMAT = {
    'mortality': '''\
Please first perform a step-by-step analysis. In your reasoning, you must integrate insights from both the structured time-series data (considering trends and abnormal values) and the unstructured clinical note (considering patient history, context, and overall status). Then, provide a final assessment of the likelihood of not surviving the hospital stay.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number between 0 and 1 representing the predicted probability of mortality (higher value means higher likelihood of death).

Example Format: ```json { "think": "The patient presents with worsening X, stable Y, and improved Z, as seen in the time-series data. The clinical note confirms the diagnosis of severe sepsis, which explains the lab trends. Given the persistent organ dysfunction mentioned in the note and the unstable vital signs, the overall assessment suggests a high risk.", "answer": 0.85 }```''',
    'readmission': '''\
Please first perform a step-by-step analysis. In your reasoning, you must integrate insights from both the structured time-series data (considering trends and abnormal values at discharge) and the unstructured clinical note (considering comorbidities, social situation, and discharge plan). Then, provide a final assessment of the likelihood of readmission within 30 days post-discharge.

Your final output must be a JSON object containing two keys:
1.  `"think"`: A string containing your detailed step-by-step clinical reasoning (under 500 words).
2.  `"answer"`: A floating-point number between 0 and 1 representing the predicted probability of readmission (higher value means higher likelihood of readmission).

Example Format: ```json { "think": "The patient's vital signs and labs stabilized before discharge. However, the discharge summary notes a complex medication regimen and limited family support. This combination of clinical stability but social vulnerability suggests a moderate risk of readmission due to potential non-adherence or lack of care at home.", "answer": 0.40 }```''',
}

EXAMPLE = {
    'mimic-iv': {
        'mortality': '''\
Input information of a patient:
The patient is a female, aged 52 years.
The patient had 4 visits that occurred at 0, 1, 2, 3.
Details of the features for each visit are as follows:
- Glascow coma scale eye opening (Unit: /. Reference range: /.): ["Spontaneously", "Spontaneously", "To pain", "To pain"]
- Heart Rate (Unit: beats per minute. Reference range: 60 - 100.): [85, 90, 115, 120]
- Temperature Fahrenheit (Unit: degrees Fahrenheit. Reference range: 97.7 - 99.5.): [98.6, 99.0, 101.5, 102.0]
...... (other features omitted for brevity)

Clinical Note:
Admission Note: 52-year-old female with a history of diabetes presented to the ED with a 2-day history of productive cough and fever. Initial workup suggestive of community-acquired pneumonia. Patient rapidly decompensated, requiring intubation for hypoxic respiratory failure. Now admitted to the ICU, started on broad-spectrum antibiotics and vasopressor support for septic shock.

Response:
```json { "think": "The patient is a 52-year-old female. The time-series data shows a clear negative trend: GCS is declining, indicating worsening neurological status. Heart rate and temperature are both increasing, consistent with a worsening systemic inflammatory response. The admission note confirms the diagnosis of severe pneumonia leading to septic shock and respiratory failure, requiring mechanical ventilation and vasopressors. The combination of deteriorating structured data and a critical diagnosis in the note indicates a very high risk of mortality.", "answer": 0.8 } ```
''',
        'readmission': '''\
Input information of a patient:
The patient is a female, aged 78 years.
The patient had 5 visits that occurred at 0, 1, 2, 3, 4.
Details of the features for each visit are as follows:
- Heart Rate (Unit: beats per minute. Reference range: 60 - 100.): [110, 105, 95, 88, 85]
- Respiratory Rate (Unit: breaths per minute. Reference range: 12 - 20.): [28, 25, 22, 18, 16]
- Sodium (Unit: mEq/L. Reference range: 136 - 145.): [132, 134, 136, 138, 138]
...... (other features omitted for brevity)

Clinical Note:
Discharge Summary: 78-year-old female admitted for CHF exacerbation. Her hospital course was complicated by hyponatremia and a mild kidney injury, which have since resolved as shown by her lab trends. She is now euvolemic. She lives alone and has a history of poor medication adherence. She will be discharged home with home health services arranged twice a week.

Response:
```json { "think": "The patient is a 78-year-old female. Her structured data shows clear improvement and stabilization before discharge: heart rate and respiratory rate have normalized, and her hyponatremia has resolved. This indicates clinical recovery from the acute CHF exacerbation. However, the discharge summary raises significant concerns: her advanced age, living alone, and a documented history of poor medication adherence are all major risk factors for readmission. While clinically stable, her social and historical context, as detailed in the note, places her at a high risk for decompensation within 30 days. The planned home health services may mitigate this, but the risk remains substantial.", "answer": 0.6 } ```
'''
    },
}

USERPROMPT = """\
I will provide you with longitudinal medical information and a clinical note for a patient. The structured data covers {LENGTH} visits that occurred at {RECORD_TIME_LIST}.
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
{DETAIL}

Clinical Note:
{NOTE}"""

UNIT = {
    'mimic-iv': 'src/structured_ehr/prompts/mimic-iv/unit.json',
}

REFERENCE_RANGE = {
    'mimic-iv': 'src/structured_ehr/prompts/mimic-iv/range.json',
}