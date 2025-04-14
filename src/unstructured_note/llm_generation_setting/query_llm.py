import json
import os
import argparse
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI
import pandas as pd
import torch
from json_repair import repair_json

from src.unstructured_note.utils.config import LLM_API_CONFIG, LMSTUDIO_MODELS_CONFIG
from src.unstructured_note.utils.classification_metrics import get_binary_metrics
from src.unstructured_note.llm_generation_setting.prompt_template import SYSTEMPROMPT, INSTRUCTION_PROMPT


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
            ]
        )
    except Exception as e:
        raise e
    return result.choices[0].message.content, result.usage.prompt_tokens, result.usage.completion_tokens


def prepare_prompt(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Prepare the prompt for the LLM based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (system prompt, instruction prompt)
    """
    # Get system prompt
    system_prompt = SYSTEMPROMPT.strip()

    # Get instruction prompt
    instruction_prompt = INSTRUCTION_PROMPT[args.task].strip()

    return system_prompt, instruction_prompt


def setup_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Set up output paths for logits and prompts.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (logits path, prompts path, performance path)
    """
    save_filename = 'output'

    if args.output_logits:
        logits_path = os.path.join(args.logits_root, 'generation', args.dataset, args.task, args.model)
        perf_path = os.path.join(args.perf_root, 'generation', args.dataset, args.task, args.model)
        os.makedirs(logits_path, exist_ok=True)
        os.makedirs(perf_path, exist_ok=True)
    else:
        logits_path = ''
        perf_path = ''

    if args.output_prompts:
        prompts_path = os.path.join(args.prompts_root, 'generation', args.dataset, args.task, args.model)
        os.makedirs(prompts_path, exist_ok=True)
    else:
        prompts_path = ''

    return logits_path, prompts_path, perf_path, save_filename


def load_dataset(args: argparse.Namespace) -> Tuple[List, List, List]:
    """
    Load dataset based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of dataset components
    """
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', f'my_datasets/{args.dataset}/processed/split')
    data = pd.read_pickle(os.path.join(dataset_path, 'test_data.pkl'))
    ids = [item['id'] for item in data]
    notes = [item['x_note'] for item in data]
    ys = [item[f'y_{args.task}'] for item in data]

    return ids, notes, ys


def process_result(result: str, y: Any) -> Tuple[Any, Any]:
    """
    Process the LLM result into prediction and get the corresponding label.

    Args:
        result: LLM result string
        y: Ground truth labels

    Returns:
        Tuple of (processed prediction, ground truth label)
    """
    # Get the label from the ground truth
    label = y[-1]

    # Parse the result into the correct format
    try:
        result_dict = repair_json(result, return_objects=True)
        if isinstance(result_dict, list):
            result_dict = result_dict[-1]
        think = result_dict['think']
        answer = result_dict['answer']
        try:
            pred = float(answer)
        except ValueError as e:
            print(f"Error converting answer to float: {answer}, error: {e}")
            pred = -1.0
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON content: {result}")

    return pred, label, think


def evaluate_binary_task(logits: Dict) -> pd.DataFrame:
    """
    Evaluate binary task performance.

    Args:
        logits: Dictionary containing labels and predictions

    Returns:
        DataFrame with performance metrics
    """
    _labels = logits['labels']
    _preds = logits['preds']

    # Calculate metrics for all samples
    _metrics = get_binary_metrics(torch.Tensor(_preds), torch.Tensor(_labels))

    # Filter out unknown samples
    labels = []
    preds = []
    for label, pred in zip(_labels, _preds):
        if pred != 0.501:
            labels.append(label)
            preds.append(pred)

    # Calculate metrics for filtered samples
    metrics = get_binary_metrics(torch.Tensor(preds), torch.Tensor(labels))

    # Prepare data for DataFrame
    data = {'count': [len(_labels), len(labels)]}
    data = dict(data, **{
        k: [f'{v1 * 100:.2f}', f'{v2 * 100:.2f}'] for k, v1, v2 in zip(
            _metrics.keys(),
            _metrics.values(),
            metrics.values()
        )
    })

    return pd.DataFrame(data=data, index=['all', 'w/o unknown'])


def run(args: argparse.Namespace):
    """
    Main function to run the LLM evaluation.

    Args:
        args: Command line arguments
    """
    prompt_tokens = 0
    completion_tokens = 0

    # Validate arguments
    assert args.dataset in ['mimic-iv'], f'Unknown dataset: {args.dataset}'
    assert args.task in ['mortality', 'readmission'], f'Unknown task: {args.task}'

    # Load the dataset
    ids, notes, ys = load_dataset(args)

    # Prepare the system prompt, instruction_prompt
    system_prompt, instruction_prompt = prepare_prompt(args)

    # Initialize LLM
    if args.output_logits:
        if args.model.lower() in ['deepseek-v3', 'deepseek-r1']:
            llm_config = LLM_API_CONFIG[f'{args.model.lower()}-ark']
        elif args.model in LMSTUDIO_MODELS_CONFIG.keys():
            llm_config = LLM_API_CONFIG['llm-studio']
            llm_config['model_name'] = LMSTUDIO_MODELS_CONFIG[args.model]['lmstudio_id']
        elif args.model.lower() == 'o3-mini-high':
            llm_config = LLM_API_CONFIG['default']
            llm_config['model_name'] = 'o3-mini-high'
        elif args.model.lower() == 'chatgpt-4o-latest':
            llm_config = LLM_API_CONFIG['v8']
            llm_config['model_name'] = 'chatgpt-4o-latest'
        else:
            raise ValueError(f'Unknown model: {args.model}')
        llm = OpenAI(api_key=llm_config['api_key'], base_url=llm_config['base_url'])

    # Setup output paths
    logits_path, prompts_path, perf_path, save_filename = setup_output_paths(args)

    # Process each patient
    labels = []
    preds = []

    for pid, note, y in tqdm(zip(ids, notes, ys), total=len(notes)):
        # Process patient ID
        if isinstance(pid, float):
            pid = str(round(pid))

        # Check if the patient has already been processed
        if os.path.exists(os.path.join(logits_path, f'{pid}.pkl')):
            print(f'Patient {pid} already processed, skipping.')
            continue

        # Create the user prompt
        user_prompt = f"{instruction_prompt}\n\nNote:\n{note}"

        # Save prompts if required
        if args.output_prompts:
            with open(os.path.join(prompts_path, f'{pid}.txt'), 'w') as f:
                f.write('System Prompt: ' + system_prompt + '\n\n')
                f.write('User Prompt: ' + user_prompt)

        # Query LLM and save results if required
        if args.output_logits:
            try:
                result, prompt_token, completion_token = query_llm(
                    model_name=llm_config['model_name'],
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
            try:
                pred, label, think = process_result(result, y)

                # Save the result
                pd.to_pickle({
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'response': result,
                    'think': think,
                    'pred': pred,
                    'label': label,
                }, os.path.join(logits_path, f'{pid}.pkl'))

                labels.append(label)
                preds.append(pred)

            except Exception as e:
                print(f'Error processing result for patient {pid}: {e}')

                # Save original result for debugging
                pd.to_pickle({
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'response': result,
                }, os.path.join(logits_path, f'{pid}.pkl'))
                continue

    if args.output_logits:
        # Save the final results
        pd.to_pickle({
            'config': vars(args),
            'preds': preds,
            'labels': labels,
        }, os.path.join(logits_path, f'0_{save_filename}.pkl'))

        # Save performance metrics
        try:
            performance_metrics = evaluate_binary_task({
                'labels': labels,
                'preds': preds,
            })
            performance_metrics.to_csv(os.path.join(perf_path, f'{save_filename}.csv'), index=False)
        except Exception as e:
            print(f'Error evaluating performance: {e}')


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run LLM evaluation on structured EHR data')

    # Dataset and task configuration
    parser.add_argument('--dataset', '-d', type=str, required=True, choices=['mimic-iv'], help='Dataset to use')
    parser.add_argument('--task', '-t', type=str, required=True, choices=['mortality', 'readmission'], help='Task to perform')
    parser.add_argument('--model', '-m', type=str, required=True, help='LLM model to use')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed for reproducibility')

    # Output configuration
    parser.add_argument('--output_logits', action='store_true', default=False,
                       help='Save model predictions')
    parser.add_argument('--output_prompts', action='store_true', default=False,
                       help='Save prompts used')
    parser.add_argument('--logits_root', type=str, default='logits',
                       help='Root directory for saving logits')
    parser.add_argument('--prompts_root', type=str, default='logs',
                       help='Root directory for saving prompts')
    parser.add_argument('--perf_root', type=str, default='performance',
                        help='Root directory for saving performance metrics')

    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print(f"Running with configuration: Model: {args.model}, Dataset: {args.dataset}, Task: {args.task}. Output {'logits and performance' if args.output_logits else 'prompts' if args.output_prompts else 'none'}.")

    # Run the evaluation
    run(args)