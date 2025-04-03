import json
import os
import argparse
from typing import List, Tuple, Any
from pathlib import Path

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI
import pandas as pd

from structured_ehr.utils.llm_configs import LLM_MODELS_SETTINGS
from structured_ehr.prompts.prompt_template import *


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def query_llm(
    model_name: str,
    llm: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> Tuple[str, int, int]:
    """
    Query the LLM with retry logic.
    
    Args:
        model_name: Name of the model to use
        llm: OpenAI client instance
        system_prompt: System prompt for the model
        user_prompt: User prompt for the model
        
    Returns:
        Tuple of (model response content, prompt tokens used, completion tokens used)
    """
    try:
        result = llm.chat.completions.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
    except Exception as e:
        raise e
    return result.choices[0].message.content, result.usage.prompt_tokens, result.usage.completion_tokens


def format_input(
    patient: List,
    dataset: str,
    features: List[str],
    mask: List[List[int]],
) -> str:
    """
    Format patient data for LLM input.
    
    Args:
        patient: List of patient visits
        dataset: Dataset name ('mimic-iv' or 'tjh')
        features: List of feature names
        mask: Missing value masks
        
    Returns:
        Formatted string with patient details
    """
    feature_values = {}
    numerical_features = ['Diastolic blood pressure', 'Fraction inspired oxygen', 'Glucose', 'Heart Rate', 'Height', 'Mean blood pressure', 'Oxygen saturation', 'Respiratory rate', 'Systolic blood pressure', 'Temperature', 'Weight', 'pH']
    categorical_features = ['Capillary refill rate', 'Glascow coma scale eye opening', 'Glascow coma scale motor response', 'Glascow coma scale total', 'Glascow coma scale verbal response']
    
    if dataset == 'mimic-iv':
        for i, feature in enumerate(features):
            if feature in numerical_features:
                feature_values[feature] = []
                for visit, m in zip(patient, mask):
                    value = str(visit[i])
                    feature_values[feature].append(value)
        for categorical_feature in categorical_features:
            indexes = [i for i, f in enumerate(
                features) if f.startswith(categorical_feature)]
            feature_values[categorical_feature] = []
            for visit in patient:
                values = [visit[i] for i in indexes]
                if 1 not in values:
                    feature_values[categorical_feature].append('unknown')
                else:
                    for i in indexes:
                        if visit[i] == 1:
                            feature_values[categorical_feature].append(
                                features[i].split('->')[-1])
                            break
        features = categorical_features + numerical_features
    elif dataset == 'tjh':
        for i, feature in enumerate(features):
            feature_values[feature] = []
            for visit, m in zip(patient, mask):
                value = str(visit[i])
                feature_values[feature].append(value)

    detail = ''
    for feature in features:
        detail += f'- {feature}: [{", ".join(feature_values[feature])}]\n'
    return detail


def prepare_prompt(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """
    Prepare the prompt for the LLM based on configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (system prompt, unit range context, example text)
    """
    # Get task description
    task_description = TASK_DESCRIPTION_AND_RESPONSE_FORMAT[args.task]
    
    # Prepare unit and reference range information if needed
    if args.unit or args.reference_range:
        unit_range = ''
        unit_values = dict(json.load(open(UNIT[args.dataset])))
        range_values = dict(json.load(open(REFERENCE_RANGE[args.dataset])))
        for feature in unit_values.keys():
            unit_range += f'- {feature}: '
            if args.unit:
                unit_range = unit_range + unit_values[feature] + ' '
            if args.reference_range:
                unit_range = unit_range + range_values[feature]
            unit_range += '\n'
    else:
        unit_range = ''
    
    # Prepare few-shot examples
    if args.n_shot == 0:
        example = ''
    elif args.n_shot == 1:
        example = f'Here is an example of input information:\n'
        example += 'Example #1:'
        example += EXAMPLE[args.dataset][args.task][0] + '\n'
    else:
        example = f'Here are {args.n_shot} examples of input information:\n'
        for i in range(args.n_shot):
            example += f'Example #{i + 1}:'
            example += EXAMPLE[args.dataset][args.task][i] + '\n'
    
    if args.prompt_engineering:
        example = COT[args.dataset]
    
    # Prepare response format
    if args.prompt_engineering:
        response_format = RESPONSE_FORMAT['cot']
    else:
        response_format = RESPONSE_FORMAT[args.task]

    return SYSTEMPROMPT[args.dataset], task_description, unit_range, example, response_format


def setup_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Set up output paths for logits and prompts.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (logits path, prompts path)
    """
    save_filename = f'{str(args.n_shot)}shot'
    if args.unit:
        save_filename += '_unit'
    if args.reference_range:
        save_filename += '_range'
    if args.prompt_engineering:
        save_filename += '_cot'
    
    if args.output_logits:
        logits_path = os.path.join(args.logits_root, args.dataset, args.task, args.model, save_filename)
        Path(logits_path).mkdir(parents=True, exist_ok=True)
    else:
        logits_path = ''
    
    if args.output_prompts:
        prompts_path = os.path.join(args.prompts_root, args.dataset, args.task, args.model, save_filename)
        Path(prompts_path).mkdir(parents=True, exist_ok=True)
    else:
        prompts_path = ''
    
    return logits_path, prompts_path, save_filename


def load_dataset(args: argparse.Namespace) -> Tuple[List, List, List, List, List, List]:
    """
    Load dataset based on configuration.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of dataset components
    """
    dataset_path = f'datasets/{args.dataset}/processed/fold_llm'
    xs = pd.read_pickle(os.path.join(dataset_path, 'test_x_no_impute.pkl'))
    ys = pd.read_pickle(os.path.join(dataset_path, 'test_y.pkl'))
    pids = pd.read_pickle(os.path.join(dataset_path, 'test_pid.pkl'))
    missing_masks = pd.read_pickle(os.path.join(
        dataset_path, 'test_x_missing_masks.pkl'))
    features = pd.read_pickle(os.path.join(
        dataset_path, 'all_features.pkl'))[2:]
    record_times = pd.read_pickle(os.path.join(
        dataset_path, 'test_x_record_times.pkl'))
    
    return xs, ys, pids, missing_masks, features, record_times


def process_result(result: str, args: argparse.Namespace, y: Any) -> Tuple[Any, Any]:
    """
    Process the LLM result into prediction and get the corresponding label.
    
    Args:
        result: LLM result string
        args: Command line arguments
        y: Ground truth labels
        
    Returns:
        Tuple of (processed prediction, ground truth label)
    """
    # Determine the correct label based on the task
    if args.task == 'outcome':
        label = y[0][0]
    elif args.task == 'readmission':
        label = y[0][2]
    else:
        raise ValueError(f'Unknown task: {args.task}')
    
    # Parse the result into the correct format
    try:
        if args.prompt_engineering:
            pred = result
        else:
            pred = float(result)
    except:
            pred = 0.501
    
    return pred, label


def run(args: argparse.Namespace):
    """
    Main function to run the LLM evaluation.
    
    Args:
        args: Command line arguments
    """
    prompt_tokens = 0
    completion_tokens = 0

    # Validate arguments
    assert args.dataset in ['tjh', 'mimic-iv'], f'Unknown dataset: {args.dataset}'
    assert args.task in ['outcome', 'readmission'], f'Unknown task: {args.task}'

    # Load the dataset
    xs, ys, pids, missing_masks, features, record_times = load_dataset(args)

    # Prepare the system prompt, unit range context, and examples
    task_description, system_prompt, unit_range, example, response_format = prepare_prompt(args)
    
    # Initialize LLM
    if 'deepseek' in args.model.lower():
        llm_config = LLM_MODELS_SETTINGS['deepseek-v3-ark']
    else:
        raise ValueError(f'Unknown model: {args.model}')
    llm = OpenAI(api_key=llm_config['api_key'], base_url=llm_config['base_url'])
    
    # Setup output paths
    sub_logits_path, sub_prompts_path, save_filename = setup_output_paths(args)
    
    # Process each patient
    labels = []
    preds = []
    
    for x, y, pid, record_time, missing_mask in tqdm(zip(xs, ys, pids, record_times, missing_masks), total=len(xs)):
        # Process patient ID
        if isinstance(pid, float):
            pid = str(round(pid))
        
        # Extract basic patient info
        length = len(x)
        sex = 'male' if x[0][0] == 1 else 'female'
        age = x[0][1]
        x = [visit[2:] for visit in x]
        
        # Format patient detail
        detail = format_input(
            patient=x,
            dataset=args.dataset,
            features=features,
            mask=missing_mask
        )
        
        # Create the user prompt
        input_format_description = INPUT_FORMAT_DESCRIPTION
        user_prompt = USERPROMPT.format(
            INPUT_FORMAT_DESCRIPTION=input_format_description,
            TASK_DESCRIPTION_AND_RESPONSE_FORMAT=task_description,
            UNIT_RANGE_CONTEXT=unit_range,
            EXAMPLE=example,
            SEX=sex,
            AGE=age,
            LENGTH=length,
            RECORD_TIME_LIST=', '.join(list(map(str, record_time))),
            DETAIL=detail,
            RESPONSE_FORMAT=response_format,
        )
        
        # Save prompts if required
        if args.output_prompts:
            with open(os.path.join(sub_prompts_path, f'{pid}.txt'), 'w') as f:
                f.write(user_prompt)
        
        # Query LLM and save results if required
        if args.output_logits:
            try:
                result, prompt_token, completion_token = query_llm(
                    model=args.model,
                    llm=llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            except Exception as e:
                print(f'Error querying LLM for patient {pid}: {e}')
                continue
            
            prompt_tokens += prompt_token
            completion_tokens += completion_token
            
            # Process the result
            pred, label = process_result(result, args, y)
            
            # Save the result
            pd.to_pickle({
                'prompt': user_prompt,
                'pred': pred,
                'label': label,
            }, os.path.join(sub_logits_path, f'{pid}.pkl'))
            
            labels.append(label)
            preds.append(pred)
    
    # Save the final results
    if args.output_logits:
        pd.to_pickle({
            'config': vars(args),
            'preds': preds,
            'labels': labels,
        }, os.path.join(sub_logits_path, save_filename + '.pkl'))


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run LLM evaluation on structured EHR data')
    
    # Dataset and task configuration
    parser.add_argument('--dataset', type=str, required=True, choices=['tjh', 'mimic-iv'], help='Dataset to use')
    parser.add_argument('--task', type=str, required=True, choices=['outcome', 'readmission'], help='Task to perform')
    parser.add_argument('--model', type=str, required=True, help='LLM model to use')
    
    # Prompt configuration
    parser.add_argument('--n_shot', type=int, default=0,
                       help='Number of examples to include in the prompt')
    parser.add_argument('--unit', action='store_true',
                       help='Include unit information in the prompt')
    parser.add_argument('--reference_range', action='store_true',
                       help='Include reference range information in the prompt')
    parser.add_argument('--prompt_engineering', action='store_true',
                       help='Use chain-of-thought prompt engineering')
    
    # Output configuration
    parser.add_argument('--output_logits', action='store_true', default=True,
                       help='Save model predictions')
    parser.add_argument('--output_prompts', action='store_true', default=False,
                       help='Save prompts used')
    parser.add_argument('--logits_root', type=str, default='logits',
                       help='Root directory for saving logits')
    parser.add_argument('--prompts_root', type=str, default='logs',
                       help='Root directory for saving prompts')
    
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    
    # Print configuration
    print(f'Running with configuration: Model: {args.model}, Dataset: {args.dataset}, Task: {args.task}')
    
    # Run the evaluation
    run(args)