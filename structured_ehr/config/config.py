OPENAI_API_KEY = ''
OPENAI_BASE_URL = ''
GOOGLE_API_KEY = ''

configs = [
    {
        'model': 'gpt-3.5-turbo',   # gpt-3.5-turbo, gpt-4-1106-preview, llama3:8b, llama3:70b
        'form': 'string',    # batches, string (, list)
        'dataset': 'mimic-iv',    # tjh, mimic-iv
        'task': 'outcome',   # outcome for tjh and mimic-iv, readmission for mimic-iv
        'time': 0,  # 0 for upon discharge, 1 for 1 month, 2 for 6 months
        'n_shot': 1,    # 0, 1...
        'unit': True,   # True, False
        'reference_range': True,    # True, False
        'impute': 1,    # 0 for no impute, 1 for impute
    }
]