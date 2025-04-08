SYSTEMPROMPT = 'You are an experienced doctor in Intensive Care Unit (ICU) treatment.'

INSTRUCTION_PROMPT = {
    "outcome": """Based on the intensive care clinical notes, please predict the patient's mortality outcome. 1 for dead, 0 for alive. The closer to 1, the more likely the patient will die. Please output the probability from 0 to 1. Please output your reasoning process and answer in the json format: { "reasoning": "your reasoning process", "answer": 0.XX }.""",
    "readmission": """Based on the intensive care clinical notes, please predict the patient's readmission probability. 1 for readmission, 0 for no readmission. The closer to 1, the more likely the patient will be readmitted. Please output the probability from 0 to 1. Please output your reasoning process and answer in the json format: { "reasoning": "your reasoning process", "answer": 0.XX }."""
}