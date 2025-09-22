import os
import json
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
from json_repair import repair_json


def _check_pred_from_dict(result_dict) -> bool:
    """
    一个辅助函数，用于从加载的结果字典中安全地提取 'pred' 值。
    它会检查 'pred' 键，如果不存在，则尝试从 'response' 键中修复并解析JSON。
    如果存在，则返回True，否则返回False。
    """
    if "pred" in result_dict:
        pred = result_dict["pred"]
    elif "response" in result_dict and isinstance(result_dict["response"], str):
        try:
            # 修复可能格式不正确的JSON字符串
            repaired_obj = repair_json(result_dict["response"])
            if isinstance(repaired_obj, dict) and "pred" in repaired_obj:
                pred = repaired_obj["pred"]
            else:
                return False
        except Exception:
            return False
    else:
        return False

    # 处理pred为NaN的情况
    if pd.isna(pred):
        return False

    return True


def extract_total_missing(data_modality, dataset, task, model, setting):
    """
    根据不同的数据模态、数据集、模型和设置，提取预测值和真实标签。

    Args:
        data_modality (str): 数据模态，可选值为 "multimodal", "unstructured_note", "structured_ehr"。
        dataset (str): 数据集名称，例如 "mimic-iv", "mimic-iii", "tjh"。
        task (str): 任务名称，例如 "mortality", "readmission", "los"。
        model (str): 模型名称，例如 "OpenBioLLM", "BERT", "AdaCare-GatorTron"。
        setting (str): 具体的实验设置。
            - 对于 "multimodal": "0shot_unit_range" (Generation模式)。
            - 对于 "unstructured_note": "prompt_setting" (Generation模式)。
            - 对于 "structured_ehr": "0shot", "0shot_unit_range", "1_shot_unit_range" (Generation模式)。

    Returns:
        tuple: 一个包含两个numpy数组的元组 (preds, labels)。
               如果找不到结果，则返回 (None, None)。
    """
    log_dir = ""
    if data_modality == "multimodal":
        log_dir = f"logs/multimodal/{dataset}/{task}/{model}/{setting}"
    elif data_modality == "unstructured_note":
        log_dir = f"logs/unstructured_note/{dataset}-note/{task}/{model}/{setting}"
    elif data_modality == "structured_ehr":
        log_dir = f"logs/structured_ehr/{dataset}-ehr/{task}/{model}/{setting}"
    else:
        raise ValueError(f"未知的数据模态: {data_modality}")

    test_data_path = f"my_datasets/{dataset}/processed/split/test_data.pkl"
    test_data = pd.read_pickle(test_data_path)
    test_pids = [item["id"] for item in test_data]
    total_missing = 0

    for pid in test_pids:
        pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
        results_path_json = os.path.join(log_dir, f"{pid_str}.json")
        results_path_pkl = os.path.join(log_dir, f"{pid_str}.pkl")

        if os.path.exists(results_path_json):
            with open(results_path_json, 'r') as f:
                results = json.load(f)
            missing = not _check_pred_from_dict(results)
        elif os.path.exists(results_path_pkl):
            results = pd.read_pickle(results_path_pkl)
            missing = not _check_pred_from_dict(results)
        else:
            missing = 1

        total_missing += missing

    return total_missing


def generate_latex_from_csv(df, output_file_path):
    """
    读取一个 DataFrame，将其转换为带有 multirow 的 LaTeX 表格代码，
    并将数值格式化为四位小数。

    Args:
        df (pd.DataFrame): 输入的 DataFrame。
        output_file_path (str): 输出文件路径。
    """
    output_str = ""

    # --- 开始生成 LaTeX 代码 ---

    # --- 逐行生成表格内容 ---

    # 找到 'modality' 和 'model' 分组的起始行索引
    model_starts = df.index[df['model'].ne(df['model'].shift())]
    model_ends = [item - 1 for item in model_starts[1:]]
    modality_starts = df.index[df['modality'].ne(df['modality'].shift())]
    modality_ends = [item - 1 for item in modality_starts[1:]]

    print(model_starts)
    print(model_ends)
    print(modality_starts)
    print(modality_ends)

    # 计算每个分组的大小，用于 \multirow
    model_counts = df['model'].value_counts()
    modality_counts = df.groupby(['model', 'modality']).size()

    # 遍历数据生成每一行
    for index, row in df.iterrows():
        print(index, row)
        line_parts = []

        # 第1列: modality
        if index in model_starts:
            count = model_counts[row['model']]
            line_parts.append(f"\\multirow{{{count}}}{{*}}{{{row['model']}}}")
        else:
            line_parts.append("")

        # 第2列: model
        if index in modality_starts:
            count = modality_counts.get((row['model'], row['modality']), 1)
            if count == 1:
                line_parts.append(row['modality'])
            else:
                line_parts.append(f"\\multirow{{{count}}}{{*}}{{{row['modality']}}}")
        else:
            line_parts.append("")

        # 第3列: setting
        line_parts.append(row['setting'])

        # 后续数值列，格式化为四位小数
        features = ['tjh_mortality', 'tjh_los', 'mimic-iv_mortality', 'mimic-iv_readmission', 'mimic-iii_mortality']
        for feature in features:
            if feature in df.columns:
                line_parts.append(f"{row[feature] * 100:.2f}" if row[feature] != "-" else "-")

        # 用 ' & ' 连接所有部分，并添加 LaTeX 换行符
        print_str = " & ".join(line_parts) + " \\\\"
        if index in model_ends:
            print_str = print_str + " \\midrule"
        elif index in modality_ends:
            print_str = print_str + " \\cmidrule{2-8}"
        output_str += print_str + "\n"

    with open(output_file_path, 'w') as f:
        f.write(output_str)


def main():
    """
    主函数，用于系统性地遍历所有实验配置，并提取结果。
    其组织结构为：数据集 -> 任务 -> 数据模态 -> 模型 -> 设置
    """
    # --- 定义所有实验的配置 ---
    # 定义数据模态和相关任务
    MODALITY_DATASET_TASKS = {
        "unstructured_note": [("mimic-iv", "mortality"), ("mimic-iv", "readmission"), ("mimic-iii", "mortality")],
        "structured_ehr": [("tjh", "mortality"), ("tjh", "los"), ("mimic-iv", "mortality"), ("mimic-iv", "readmission")],
        "multimodal": [("mimic-iv", "mortality"), ("mimic-iv", "readmission")]
    }

    # 定义不同模态下的模型和设置
    GENERATION_MODELS = ["OpenBioLLM", "Qwen2.5-7B", "Gemma-3-4B", "deepseek-v3-chat", "chatgpt-4o-latest", "HuatuoGPT-o1-7B", "DeepSeek-R1-7B", "deepseek-v3-reasoner", "deepseek-r1", "o3-mini-high", "gpt-5-chat-latest"]

    # 定义属性对应的列名
    MODEL_NAME_MAPPING = {
        "OpenBioLLM": "OpenBioLLM-8B",
        "deepseek-v3-chat": "DeepSeek-V3.1",
        "deepseek-v3-reasoner": "DeepSeek-V3.1-Reasoning",
        "deepseek-r1": "DeepSeek-R1",
        "chatgpt-4o-latest": "GPT-4o",
        "gpt-5-chat-latest": "GPT-5",
    }
    SETTING_MAPPING = {
        "0shot": "base prompt",
        "0shot_unit_range": "optimized prompt",
        "1shot_unit_range": "opt.+ICL",
        "prompt_setting": "prompt"
    }
    MODALITY_MAPPING = {
        "unstructured_note": "Unstructured Note",
        "multimodal": "Multimodal",
        "structured_ehr": "Structured EHR",
    }
    df_index = 0
    all_missing_df = pd.DataFrame()

    for model in GENERATION_MODELS:
        print(f"==========================================")
        print(f"Model: {model}")
        print(f"==========================================")

        model_missing_df = pd.DataFrame()

        for modality, dataset_tasks in MODALITY_DATASET_TASKS.items():
            modality_missing_df = pd.DataFrame()

            if modality == "unstructured_note":
                settings = ["prompt_setting"]
            elif modality == "multimodal":
                settings = ["0shot_unit_range"]
            elif modality == "structured_ehr":
                settings = ["0shot", "0shot_unit_range", "1shot_unit_range"]

            for setting in settings:
                setting_missing_df = pd.DataFrame()
                dataset_task_missing_dict = {
                    "model": MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model,
                    "modality": MODALITY_MAPPING[modality],
                    "setting": SETTING_MAPPING[setting],
                    "tjh_mortality": "-",
                    "tjh_los": "-",
                    "mimic-iv_mortality": "-",
                    "mimic-iv_readmission": "-",
                    "mimic-iii_mortality": "-",
                }

                for dataset, task in dataset_tasks:
                    print(f"==========================================")
                    print(f"Dataset: {dataset.upper()}, Task: {task.upper()}")
                    print(f"==========================================")

                    total_missing = extract_total_missing(modality, dataset, task, model, setting)
                    dataset_task_missing_dict[f"{dataset}_{task}"] = 1.0 * total_missing / 200
                    setting_missing_df = pd.DataFrame(dataset_task_missing_dict, index=[df_index])
                    df_index += 1

                modality_missing_df = pd.concat([modality_missing_df, setting_missing_df], axis=0)

            model_missing_df = pd.concat([model_missing_df, modality_missing_df], axis=0)

        all_missing_df = pd.concat([all_missing_df, model_missing_df], axis=0)
    os.makedirs(f"logs/error_handle", exist_ok=True)
    all_missing_df = all_missing_df.reset_index(drop=True)
    all_missing_df.to_csv(f"logs/error_handle/missing_rates.csv", index=False)
    generate_latex_from_csv(all_missing_df, f"logs/error_handle/missing_rates.txt")

if __name__ == "__main__":
    main()