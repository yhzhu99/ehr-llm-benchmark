import json
import os
import argparse
from typing import List, Tuple, Any

import pandas as pd
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from openai import OpenAI
import tiktoken
from json_repair import repair_json

from src.structured_ehr.utils.llm_configs import LLM_API_CONFIG, MODELS_CONFIG
from src.multimodal.prompts.prompt_template import SYSTEMPROMPT, USERPROMPT, UNIT, REFERENCE_RANGE, TASK_DESCRIPTION, RESPONSE_FORMAT, EXAMPLE


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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except Exception as e:
        raise e
    return result.choices[0].message.content, result.usage.prompt_tokens, result.usage.completion_tokens


def format_input(
    patient: List,
    dataset: str,
    features: List[str],
    mask: List[List[int]],
    unit: bool = False,
    reference_range: bool = False,
) -> str:
    """
    Format patient data for LLM input.

    Args:
        patient: List of patient visits
        dataset: Dataset name ("mimic-iv" or "tjh")
        features: List of feature names
        mask: Missing value masks

    Returns:
        Formatted string with patient details
    """
    feature_values = {}

    # Define some categorical features with their possible values
    categorical_features_dict = {
        "Glascow coma scale eye opening": {
            1: "No Response",
            2: "To Pain",
            3: "To Speech",
            4: "Spontaneously",
        },
        "Glascow coma scale motor response": {
            1: "No Response",
            2: "Abnormal Extension",
            3: "Abnormal Flexion",
            4: "Flex-withdraws",
            5: "Localizes Pain",
            6: "Obeys Commands",
        },
        "Glascow coma scale verbal response": {
            1: "No Response",
            2: "Incomprehensible sounds",
            3: "Inappropriate Words",
            4: "Confused",
            5: "Oriented",
        },
    }

    if dataset == "mimic-iv":
        for i, feature in enumerate(features):
            feature_values[feature] = []
            for visit, m in zip(patient, mask):
                if m[i] == 1:
                    value = "NaN"
                elif feature in categorical_features_dict.keys():
                    value = categorical_features_dict[feature].get(visit[i], "NaN")
                else:
                    value = str(visit[i])
                feature_values[feature].append(value)
    elif dataset == "tjh":
        for i, feature in enumerate(features):
            feature_values[feature] = []
            for visit, m in zip(patient, mask):
                value = str(visit[i]) if m[i] == 0 else "NaN"
                feature_values[feature].append(value)

    # Prepare unit and reference range information if needed
    unit_values = dict(json.load(open(UNIT[args.dataset])))
    range_values = dict(json.load(open(REFERENCE_RANGE[args.dataset])))

    detail = ""
    for feature in features:
        unit_range = ""
        if unit or reference_range:
            unit_range = " ("
            if unit:
                unit_range += f"{unit_values[feature]} "
            if reference_range:
                unit_range += range_values[feature]
            unit_range = unit_range.rstrip() + ")"
        detail += f"- {feature}{unit_range}: [{', '.join(feature_values[feature])}]\n"
    return detail.strip()


def prepare_prompt(args: argparse.Namespace) -> Tuple[str, str, str, str]:
    """
    Prepare the prompt for the LLM based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (system prompt, unit range context, example text)
    """
    # Get system prompt
    system_prompt = SYSTEMPROMPT[args.dataset].strip()

    # Get task description
    task_description = TASK_DESCRIPTION[args.task].strip()

    # Prepare few-shot examples
    if args.n_shot == 0:
        example = ""
    elif args.n_shot == 1:
        example = "Example:\n"
        example += EXAMPLE[args.dataset][args.task]
    else:
        raise ValueError(f"Invalid n_shot value: {args.n_shot}")
    example = example.strip()

    # Get the response format
    response_format = RESPONSE_FORMAT[args.task].strip()

    return system_prompt, task_description, example, response_format


def setup_output_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """
    Set up output paths for logits and prompts.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (logits path, prompts path)
    """
    save_filename = f"{str(args.n_shot)}shot"
    if args.unit:
        save_filename += "_unit"
    if args.reference_range:
        save_filename += "_range"

    if args.output_logits:
        logits_path = os.path.join(args.logits_root, "multimodal", args.dataset, args.task, args.model, save_filename)
        os.makedirs(logits_path, exist_ok=True)
    else:
        logits_path = ""

    if args.output_prompts:
        prompts_path = os.path.join(args.prompts_root, "multimodal", args.dataset, args.task, args.model, save_filename)
        os.makedirs(prompts_path, exist_ok=True)
    else:
        prompts_path = ""

    return logits_path, prompts_path, save_filename


def load_dataset(args: argparse.Namespace) -> Tuple[List, List, List, List, List, List, Any]:
    """
    Load dataset based on configuration.

    Args:
        args: Command line arguments

    Returns:
        Tuple of dataset components
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", f"my_datasets/{args.dataset}/processed/split")
    data = pd.read_pickle(os.path.join(dataset_path, "test_data.pkl"))
    ids = [item["id"] for item in data]
    xs = [item["x_llm_ts"] for item in data]
    notes = [item["x_note"] for item in data]
    ys = [item[f"y_{args.task}"] for item in data]
    missing_masks = [item["missing_mask"] for item in data]
    record_times = [item["record_time"] for item in data]
    labtest_features = pd.read_pickle(os.path.join(dataset_path, "labtest_features.pkl"))
    if args.task == "los":
        try:
            los_info = pd.read_pickle(os.path.join(dataset_path, "los_info.pkl"))
        except FileNotFoundError:
            raise FileNotFoundError(f"LOS info file not found in {dataset_path}.")
    else:
        los_info = None

    return ids, xs, notes, ys, missing_masks, record_times, labtest_features, los_info


def process_result(result: str, y: Any) -> Tuple[float, float, str]:
    """
    Process the LLM result into prediction and get the corresponding label.

    Args:
        result: LLM result string
        y: Ground truth labels

    Returns:
        Tuple of (processed prediction, ground truth label)
    """
    # Get the label from the ground truth
    label = y[0] if isinstance(y, list) else y

    # Parse the result into the correct format
    try:
        result_dict = repair_json(result, return_objects=True)
        if isinstance(result_dict, list):
            result_dict = result_dict[-1]
        think = result_dict["think"]
        answer = result_dict["answer"]
        try:
            pred = float(answer)
        except ValueError as e:
            print(f"Error converting answer to float: {answer}, error: {e}")
            pred = -1.0
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON content: {result}")

    return pred, label, think


def run(args: argparse.Namespace):
    """
    Main function to run the LLM evaluation.

    Args:
        args: Command line arguments
    """
    prompt_tokens = 0
    completion_tokens = 0

    # Validate arguments
    assert args.dataset in ["tjh", "mimic-iv"], f"Unknown dataset: {args.dataset}"
    assert args.task in ["mortality", "los", "readmission"], f"Unknown task: {args.task}"
    if args.task == "readmission":
        assert args.dataset == "mimic-iv", "Readmission task is only available for MIMIC-IV dataset"
    elif args.task == "los":
        assert args.dataset == "tjh", "LOS task is only available for TJH dataset"

    # Load the dataset
    ids, xs, notes, ys, missing_masks, record_times, features, _ = load_dataset(args)

    # Prepare the system prompt, unit range context, and examples
    system_prompt, task_description, example, response_format = prepare_prompt(args)

    # Initialize LLM
    llm_config = LLM_API_CONFIG["laozhang"]
    if args.model.lower() == "deepseek-v3-chat":
        llm_config = LLM_API_CONFIG["deepseek-v3-chat"]
    elif args.model.lower() == "deepseek-v3-reasoner":
        llm_config = LLM_API_CONFIG["deepseek-v3-reasoner"]
    elif args.model.lower() == "deepseek-r1":
        llm_config["model_name"] = "deepseek-r1-250528"
    elif args.model.lower() == "o3-mini-high":
        llm_config["model_name"] = "o3-mini"
        llm_config["reasoning_effort"] = "high"
    elif args.model.lower() == "chatgpt-4o-latest":
        llm_config["model_name"] = "chatgpt-4o-latest"
    elif args.model.lower() == "gpt-5-chat-latest":
        llm_config["model_name"] = "gpt-5-chat-latest"
        llm_config["reasoning_effort"] = "high"
    elif args.model in MODELS_CONFIG.keys():
        llm_config = LLM_API_CONFIG["llm-studio"]
        llm_config["model_name"] = MODELS_CONFIG[args.model]["lmstudio_id"]
    else:
        raise ValueError(f"Unknown model: {args.model}")
    llm = OpenAI(api_key=llm_config["api_key"], base_url=llm_config["base_url"])

    # Setup output paths
    logits_path, prompts_path, save_filename = setup_output_paths(args)

    # Process each patient
    labels = []
    preds = []

    for pid, x, note, y, missing_mask, record_time in tqdm(zip(ids, xs, notes, ys, missing_masks, record_times), total=len(xs)):
        # Process patient ID
        if isinstance(pid, float):
            pid = str(round(pid))

        # Check if the patient has already been processed
        if os.path.exists(os.path.join(logits_path, f"{pid}.json")):
            print(f"Patient {pid} already processed, skipping.")
            continue

        # Extract basic patient info
        length = len(x)
        sex = "male" if x[0][0] == 1 else "female"
        age = x[0][1]
        x = [visit[2:] for visit in x]
        missing_mask = [mask[2:] for mask in missing_mask]

        # Truncate the note
        encoding = tiktoken.encoding_for_model("gpt-4o-latest")
        note = encoding.decode(encoding.encode(note)[:args.max_length])

        # Format patient detail
        detail = format_input(
            patient=x,
            dataset=args.dataset,
            features=features,
            mask=missing_mask,
            unit=args.unit,
            reference_range=args.reference_range,
        )

        # Create the user prompt
        user_prompt = USERPROMPT.format(
            TASK_DESCRIPTION=task_description,
            EXAMPLE=example,
            SEX=sex,
            AGE=age,
            LENGTH=length,
            RECORD_TIME_LIST=", ".join(list(map(str, record_time))),
            DETAIL=detail,
            RESPONSE_FORMAT=response_format,
            NOTE=note,
        )

        # Save prompts if required
        if args.output_prompts:
            with open(os.path.join(prompts_path, f"{pid}.txt"), "w") as f:
                f.write("System Prompt: " + system_prompt + "\n\n")
                f.write("User Prompt: " + user_prompt)

        # Query LLM and save results if required
        if args.output_logits:
            try:
                result, prompt_token, completion_token = query_llm(
                    model_name=llm_config["model_name"],
                    llm=llm,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            except Exception as e:
                print(f"Error querying LLM for patient {pid}: {e}")
                continue

            prompt_tokens += prompt_token
            completion_tokens += completion_token

            # Process the result
            try:
                pred, label, think = process_result(result, y)

                if pred < 0:
                    pred = 0.501 if args.task in ["mortality", "readmission"] else 0.0

                print(f"Processed result for patient {pid}: {pred}, {label}, {think[:100]}")

                # Save the result
                json.dump({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": result,
                    "think": think,
                    "pred": pred,
                    "label": label,
                }, open(os.path.join(logits_path, f"{pid}.json"), "w"), indent=4, ensure_ascii=False)

                labels.append(label)
                preds.append(pred)

            except Exception as e:
                print(f"Error processing result for patient {pid}: {e}")

                # Save original result for debugging
                json.dump({
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "response": result,
                    "label": label,
                }, open(os.path.join(logits_path, f"{pid}.json"), "w"), indent=4, ensure_ascii=False)

                print(f"Saved original result for patient {pid}")

                continue

    if args.output_logits:
        # Save the final results
        print(f"Saving final results...")

        json.dump({
            "config": vars(args),
            "preds": preds,
            "labels": labels,
        }, open(os.path.join(logits_path, f"0_{save_filename}.json"), "w"), indent=4, ensure_ascii=False)


def parse_args():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run LLM evaluation on multimodal EHR data")

    # Dataset and task configuration
    parser.add_argument("--dataset", "-d", type=str, required=True, choices=["mimic-iv"], help="Dataset to use")
    parser.add_argument("--task", "-t", type=str, required=True, choices=["mortality", "readmission"], help="Task to perform")
    parser.add_argument("--model", "-m", type=str, required=True, help="LLM model to use")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")

    # Prompt configuration
    parser.add_argument("--n_shot", "-n", type=int, default=0, choices=[0, 1],
                       help="Number of examples to include in the prompt")
    parser.add_argument("--unit", "-u", action="store_true",
                       help="Include unit information in the prompt")
    parser.add_argument("--reference_range", "-r", action="store_true",
                       help="Include reference range information in the prompt")
    parser.add_argument("--max_length", "-l", type=int, default=512, help="Maximum length of the note")

    # Output configuration
    parser.add_argument("--output_logits", action="store_true", default=False,
                       help="Save model predictions")
    parser.add_argument("--output_prompts", action="store_true", default=False,
                       help="Save prompts used")
    parser.add_argument("--logits_root", type=str, default="logs",
                       help="Root directory for saving logits")
    parser.add_argument("--prompts_root", type=str, default="logs",
                       help="Root directory for saving prompts")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Print configuration
    print(f"Running with configuration: Model: {args.model}, Dataset: {args.dataset}, Task: {args.task}. Output {'logits and performance' if args.output_logits else 'prompts' if args.output_prompts else 'none'}.")

    # Run the evaluation
    run(args)