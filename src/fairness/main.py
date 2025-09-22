import os
import json
from typing import Dict, Optional

import torch
import numpy as np
import pandas as pd
from json_repair import repair_json


def _extract_pred_from_dict(result_dict, default_value=0.501):
    """
    一个辅助函数，用于从加载的结果字典中安全地提取 'pred' 值。
    它会检查 'pred' 键，如果不存在，则尝试从 'response' 键中修复并解析JSON。
    """
    if "pred" in result_dict:
        pred = result_dict['pred']
    elif "response" in result_dict and isinstance(result_dict["response"], str):
        try:
            # 修复可能格式不正确的JSON字符串
            repaired_obj = repair_json(result_dict["response"])
            if isinstance(repaired_obj, dict) and "pred" in repaired_obj:
                pred = repaired_obj["pred"]
            else:
                pred = default_value  # 默认值
        except Exception:
            pred = default_value # 默认值
    else:
        pred = default_value # 默认值

    # 处理pred为NaN的情况
    if pd.isna(pred):
        pred = default_value

    return pred


def extract_predictions_and_labels(data_modality, dataset, task, model, setting):
    """
    根据不同的数据模态、数据集、模型和设置，提取预测值和真实标签。

    Args:
        data_modality (str): 数据模态，可选值为 "multimodal", "unstructured_note", "structured_ehr"。
        dataset (str): 数据集名称，例如 "mimic-iv", "mimic-iii", "tjh"。
        task (str): 任务名称，例如 "mortality", "readmission", "los"。
        model (str): 模型名称，例如 "OpenBioLLM", "BERT", "AdaCare-GatorTron"。
        setting (str): 具体的实验设置。
            - 对于 "multimodal": "0shot_unit_range" (Generation模式) 或 "add", "concat" (Tuning模式)。
            - 对于 "unstructured_note": "prompt_setting" (Generation模式) 或 "freeze_setting", "finetune_setting" (Tuning模式)。
            - 对于 "structured_ehr": "0shot", "0shot_unit_range" 等 (Generation模式)。

    Returns:
        tuple: 一个包含两个numpy数组的元组 (preds, labels)。
               如果找不到结果，则返回 (None, None)。
    """
    # 1. 根据参数构建日志目录
    log_dir = ""
    if data_modality == "multimodal":
        # 对于Tuning模式，实际路径中包含了"fusion_training"
        if setting in ["add", "concat", "attention", "cross_attention"]:
             log_dir = f"logs/multimodal/{dataset}/{task}/{model}/fusion_training/{setting}"
        else:
             log_dir = f"logs/multimodal/{dataset}/{task}/{model}/{setting}"
    elif data_modality == "unstructured_note":
        log_dir = f"logs/unstructured_note/{dataset}-note/{task}/{model}/{setting}"
    elif data_modality == "structured_ehr":
        log_dir = f"logs/structured_ehr/{dataset}-ehr/{task}/{model}/{setting}"
        if setting in ["few", "full"]:
            log_dir = f"logs/structured_ehr/{dataset}-ehr/{task}/dl_models_new/{model}/{setting}"
    else:
        raise ValueError(f"未知的数据模态: {data_modality}")

    # 2. 判断是Tuning模式还是Generation模式
    # Tuning模式: 结果在一个文件中 (test_results.pkl)
    is_tuning_mode = (
        (data_modality == "unstructured_note" and setting in ["freeze_setting", "finetune_setting"]) or
        (data_modality == "multimodal" and setting in ["add", "concat", "attention", "cross_attention"]) or
        (data_modality == "structured_ehr" and setting in ["few", "full"])
    )

    preds = []
    labels = []

    if is_tuning_mode:
        if data_modality == "structured_ehr":
            results_path = os.path.join(log_dir, "outputs.pkl")
            preds_key = "preds"
            labels_key = "labels"
        else:
            results_path = os.path.join(log_dir, "test_results.pkl")
            preds_key = "y_pred"
            labels_key = "y_true"
        if not os.path.exists(results_path):
            return preds, labels

        results = pd.read_pickle(results_path)
        preds = results[preds_key]
        labels = results[labels_key]

        if isinstance(preds, list):
            preds = np.array(preds)
        elif isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        if isinstance(labels, list):
            labels = np.array(labels)
        elif isinstance(labels, torch.Tensor):
            labels = labels.numpy()

    # Generation/Prompting模式: 每个样本一个结果文件
    else:
        test_data_path = f"my_datasets/{dataset}/processed/split/test_data.pkl"
        if not os.path.exists(test_data_path):
            return preds, labels

        test_data = pd.read_pickle(test_data_path)
        test_pids = [item["id"] for item in test_data]
        test_ys = [item[f'y_{task}'] for item in test_data]

        for pid, y in zip(test_pids, test_ys):
            pid_str = str(int(pid)) if isinstance(pid, float) else str(pid)
            results_path_json = os.path.join(log_dir, f"{pid_str}.json")
            results_path_pkl = os.path.join(log_dir, f"{pid_str}.pkl")

            results = {}
            if os.path.exists(results_path_json):
                with open(results_path_json, 'r') as f:
                    results = json.load(f)
                pred = _extract_pred_from_dict(results)
            elif os.path.exists(results_path_pkl):
                results = pd.read_pickle(results_path_pkl)
                pred = _extract_pred_from_dict(results)
            else:
                pred = 0.501

            preds.append(pred)
            # 处理标签可能是列表的情况
            labels.append(y[0] if isinstance(y, list) else y)

        preds = np.array(preds)
        labels = np.array(labels)

    # 3. 特殊后处理
    # 对于 structured_ehr 的 los 任务，需要对预测值进行标准化
    if data_modality == "structured_ehr" and task == "los":
        los_info_path = f"my_datasets/{dataset}/processed/split/los_info.pkl"
        if os.path.exists(los_info_path):
            los_info = pd.read_pickle(los_info_path)
            los_mean = los_info["los_mean"]
            los_std = los_info["los_std"]
            preds = [(item - los_mean) / (los_std + 1e-12) for item in preds]
        else:
            print(f"警告: LOS信息文件未找到: {los_info_path}。预测值未被标准化。")

    return preds, labels


def extract_sensitive_attributes(dataset, attribute):
    if dataset == "mimic-iv":
        data = pd.read_pickle(f"my_datasets/{dataset}/processed/fairness/test_data.pkl")
        sensitive_attributes = [item[attribute] for item in data]
        if attribute == "race":
            sensitive_attributes = np.array([1 if "white" in x.lower() else 0 for x in sensitive_attributes])
        elif attribute == "age":
            sensitive_attributes = np.array([1 if x > 50 else 0 for x in sensitive_attributes])
        sensitive_attributes = np.array(sensitive_attributes)
    elif dataset == "mimic-iii":
        data = pd.read_pickle(f"my_datasets/{dataset}/processed/fairness/test_data.pkl")
        sensitive_attributes = [item[attribute] for item in data]
        sensitive_attributes = np.array(sensitive_attributes)
    elif dataset == "tjh":
        data = [item['x_llm_ts'][0] for item in pd.read_pickle(f"my_datasets/{dataset}/processed/split/test_data.pkl")]
        if attribute == "age":
            ind = 1
        elif attribute == "gender":
            ind = 0
        else:
            raise ValueError(f"Unsupported attribute: {attribute} for dataset: {dataset}")
        sensitive_attributes = [item[ind] for item in data]
        if attribute == "age":
            sensitive_attributes = np.array([1 if x > 50 else 0 for x in sensitive_attributes])
        sensitive_attributes = np.array(sensitive_attributes)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return sensitive_attributes


def calculate_fairness_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    sensitive_attributes: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算并返回一组关于二分类模型预测的公平性指标。

    该函数通过比较受保护群体（sensitive_attribute == 1）和
    非受保护群体（sensitive_attribute == 0）之间的统计差异来评估模型的公平性。

    Args:
        true_labels (np.ndarray): 包含真实标签（0或1）的一维数组。
        pred_labels (np.ndarray): 包含模型预测标签（0或1）的一维数组。
        sensitive_attributes (np.ndarray): 包含敏感属性（0或1）的一维数组。
                                           0 代表非受保护群体, 1 代表受保护群体。
        sample_weights (Optional[np.ndarray]): 样本权重数组。如果为 None，则所有样本权重视为1。

    Returns:
        Dict[str, float]: 一个包含四个公平性指标的字典：
        - 'DI' (Disparate Impact): 歧视性影响比例。理想值为1.0。
                                   计算方式: P(Ŷ=1|A=1) / P(Ŷ=1|A=0)
        - 'SPD' (Statistical Parity Difference): 统计均等差异。理想值为0.0。
                                   计算方式: P(Ŷ=1|A=1) - P(Ŷ=1|A=0)
        - 'AOD' (Average Odds Difference): 平均机会差异。理想值为0.0。
                                   计算方式: 0.5 * [(FPR_prot - FPR_unprot) + (TPR_prot - TPR_unprot)]
        - 'EOD' (Equal Opportunity Difference): 均等机会差异。理想值为0.0。
                                   计算方式: TPR_prot - TPR_unprot
    """
    # 确保所有输入都是numpy数组
    true_labels = np.asarray(true_labels).squeeze()
    pred_labels = np.asarray(pred_labels).squeeze() > 0.5
    sensitive_attributes = np.asarray(sensitive_attributes).squeeze()

    # 如果没有提供样本权重，则所有样本权重为1
    if sample_weights is None:
        sample_weights = np.ones_like(true_labels)
    else:
        sample_weights = np.asarray(sample_weights)

    # 为受保护群体和非受保护群体创建布尔掩码
    protected_mask = (sensitive_attributes == 1)
    unprotected_mask = (sensitive_attributes == 0)

    # --- 计算基础加权统计量 ---
    # 总权重
    w_prot = np.sum(sample_weights[protected_mask])
    w_unprot = np.sum(sample_weights[unprotected_mask])

    # 预测为正例(favorable)的权重
    w_pred_pos_prot = np.sum(sample_weights[protected_mask & (pred_labels == 1)])
    w_pred_pos_unprot = np.sum(sample_weights[unprotected_mask & (pred_labels == 1)])

    # 真实为正例(favorable)的权重
    w_true_pos_prot = np.sum(sample_weights[protected_mask & (true_labels == 1)])
    w_true_pos_unprot = np.sum(sample_weights[unprotected_mask & (true_labels == 1)])

    # 真实为负例(unfavorable)的权重
    w_true_neg_prot = np.sum(sample_weights[protected_mask & (true_labels == 0)])
    w_true_neg_unprot = np.sum(sample_weights[unprotected_mask & (true_labels == 0)])

    # 真阳性(TP)和假阳性(FP)的权重
    w_tp_prot = np.sum(sample_weights[protected_mask & (pred_labels == 1) & (true_labels == 1)])
    w_tp_unprot = np.sum(sample_weights[unprotected_mask & (pred_labels == 1) & (true_labels == 1)])

    w_fp_prot = np.sum(sample_weights[protected_mask & (pred_labels == 1) & (true_labels == 0)])
    w_fp_unprot = np.sum(sample_weights[unprotected_mask & (pred_labels == 1) & (true_labels == 0)])

    # --- 计算比率 ---
    # 为避免除以零，添加一个极小值epsilon
    epsilon = 1e-8

    # 预测为正例的比率
    rate_pred_pos_prot = w_pred_pos_prot / (w_prot + epsilon)
    rate_pred_pos_unprot = w_pred_pos_unprot / (w_unprot + epsilon)

    # 真阳性率 (TPR)
    tpr_prot = w_tp_prot / (w_true_pos_prot + epsilon)
    tpr_unprot = w_tp_unprot / (w_true_pos_unprot + epsilon)

    # 假阳性率 (FPR)
    fpr_prot = w_fp_prot / (w_true_neg_prot + epsilon)
    fpr_unprot = w_fp_unprot / (w_true_neg_unprot + epsilon)

    # --- 计算最终公平性指标 ---
    di = rate_pred_pos_prot / (rate_pred_pos_unprot + epsilon)
    spd = rate_pred_pos_prot - rate_pred_pos_unprot
    eod = tpr_prot - tpr_unprot
    aod = 0.5 * ((fpr_prot - fpr_unprot) + (tpr_prot - tpr_unprot))

    return {"DI": di, "SPD": spd, "AOD": aod, "EOD": eod}



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

    # 定义表格的列格式 (l: 左对齐, c: 居中对齐)
    # 根据您的表格，前4列为文本，后4列为数值
    num_columns = len(df.columns)
    text_cols = 4
    numeric_cols = num_columns - text_cols

    # --- 逐行生成表格内容 ---

    # 找到 'modality' 和 'model' 分组的起始行索引
    modality_starts = df.index[df['modality'].ne(df['modality'].shift())]
    modality_ends = [item - 1 for item in modality_starts[1:]]
    model_starts = df.index[df['model'].ne(df['model'].shift())]
    model_ends = [item - 1 for item in model_starts[1:]]

    # 计算每个分组的大小，用于 \multirow
    modality_counts = df['modality'].value_counts()
    model_counts = df.groupby(['modality', 'model']).size()

    # 遍历数据生成每一行
    for index, row in df.iterrows():
        line_parts = []

        # 第1列: modality
        if index in modality_starts:
            count = modality_counts[row['modality']]
            line_parts.append(f"\\multirow{{{count}}}{{*}}{{{row['modality']}}}")
        else:
            line_parts.append("")

        # 第2列: model
        if index in model_starts:
            count = model_counts.get((row['modality'], row['model']), 1)
            line_parts.append(f"\\multirow{{{count}}}{{*}}{{{row['model']}}}")
        else:
            line_parts.append("")

        # 第3列: setting
        line_parts.append(row['setting'])

        # 后续数值列，格式化为四位小数
        features = ['Age', 'Gender', 'Race']
        numeric_columns = ['DI', 'SPD', 'AOD', 'EOD']
        for feature in features:
            for col in numeric_columns:
                if f"{feature}_{col}" in df.columns:
                    line_parts.append(f"{row[f'{feature}_{col}']:.4f}")

        # 用 ' & ' 连接所有部分，并添加 LaTeX 换行符
        print_str = " & ".join(line_parts) + " \\\\"
        if index in modality_ends:
            print_str = print_str + " \\midrule"
        elif index in model_ends:
            print_str = print_str + " \\cmidrule{2-15}"
        output_str += print_str + "\n"

    with open(output_file_path, 'w') as f:
        f.write(output_str)


def main():
    """
    主函数，用于系统性地遍历所有实验配置，并提取结果。
    其组织结构为：数据集 -> 任务 -> 数据模态 -> 模型 -> 设置
    """
    # --- 定义所有实验的配置 ---
    # 定义数据集和相关任务
    DATASET_TASKS = {
        "mimic-iv": ["mortality", "readmission"],
        "mimic-iii": ["mortality"],
        "tjh": ["mortality", "los"]
    }

    # 定义不同模态下的模型和设置
    MULTIMODAL_GENERATION_MODELS = ["OpenBioLLM", "Qwen2.5-7B", "Gemma-3-4B", "deepseek-v3-chat", "chatgpt-4o-latest", "HuatuoGPT-o1-7B", "DeepSeek-R1-7B", "deepseek-v3-reasoner", "deepseek-r1", "o3-mini-high", "gpt-5-chat-latest"]
    MULTIMODAL_TUNING_CONFIG = {
        ("mimic-iv", "mortality"): "AdaCare-GatorTron",
        ("mimic-iv", "readmission"): "LSTM-HuatuoGPT-o1-7B"
    }

    UNSTRUCTURED_NOTE_MODELS = ["BERT", "Clinical-Longformer", "BioBERT", "GatorTron", "ClinicalBERT", "GPT-2", "BioGPT", "meditron", "BioMistral", "OpenBioLLM", "Qwen2.5-7B", "Gemma-3-4B", "deepseek-v3-chat", "chatgpt-4o-latest", "HuatuoGPT-o1-7B", "DeepSeek-R1-7B", "deepseek-v3-reasoner", "deepseek-r1", "o3-mini-high", "gpt-5-chat-latest"]

    STRUCTURED_EHR_MODELS = ["OpenBioLLM", "Qwen2.5-7B", "Gemma-3-4B", "deepseek-v3-chat", "chatgpt-4o-latest", "HuatuoGPT-o1-7B", "DeepSeek-R1-7B", "deepseek-r1", "deepseek-v3-reasoner", "o3-mini-high", "gpt-5-chat-latest"]

    STRUCTURED_DL_EHR_MODELS = ["CatBoost", "DT", "RF", "XGBoost", "GRU", "LSTM", "Transformer", "RNN", "AdaCare", "AICare", "ConCare", "GRASP"]

    SENSITIVE_ATTRIBUTES = {
        "mimic-iv": ["Age", "Gender", "Race"],
        "mimic-iii": ["Age","Gender", "Race"],
        "tjh": ["Age","Gender"]
    }

    # 定义属性对应的列名
    MODEL_NAME_MAPPING = {
        "OpenBioLLM": "OpenBioLLM-8B",
        "gemma-3-4b-pt": "Gemma-3-4B",
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
        "add": "Add",
        "concat": "Concat",
        "attention": "Attention",
        "cross_attention": "Cross Attention",
        "freeze_setting": "freeze",
        "finetune_setting": "finetune",
        "prompt_setting": "prompt",
        "few": "10 shot",
        "full": "full shot",
    }


    for dataset, tasks in DATASET_TASKS.items():
        print(f"==========================================")
        print(f"Dataset: {dataset.upper()}")
        print(f"==========================================")

        df_index = 0

        for task in tasks:
            print(f"\n--- Task: {task} ---")
            fairness_df = pd.DataFrame()

            # 1. 处理 Unstructured Note 数据
            if dataset in ["mimic-iii", "mimic-iv"]:
                for model in UNSTRUCTURED_NOTE_MODELS:
                    for setting in ["freeze_setting", "finetune_setting", "prompt_setting"]:
                        preds, labels = extract_predictions_and_labels("unstructured_note", dataset, task, model, setting)
                        if len(preds) == 0 or len(labels) == 0:
                            continue
                        model_name = MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model
                        setting_name = SETTING_MAPPING[setting]
                        sub_dict = {
                            "modality": "Unstructured Note",
                            "model": model_name,
                            "setting": setting_name,
                        }
                        for sensitive_attribute in SENSITIVE_ATTRIBUTES[dataset]:
                            sensitive_attributes = extract_sensitive_attributes(dataset, sensitive_attribute.lower())
                            fairness_metrics = calculate_fairness_metrics(labels, preds, sensitive_attributes)
                            for key in fairness_metrics:
                                sub_dict[f"{sensitive_attribute}_{key}"] = fairness_metrics[key]
                        sub_df = pd.DataFrame(sub_dict, index=[df_index])
                        fairness_df = pd.concat([fairness_df, sub_df], axis=0)
                        df_index += 1

            # 2. 处理 Structured EHR 数据
            if dataset in ["tjh", "mimic-iv"]:
                for model in STRUCTURED_DL_EHR_MODELS:
                    for setting in ["few", "full"]:
                        preds, labels = extract_predictions_and_labels("structured_ehr", dataset, task, model, setting)
                        if len(preds) == 0 or len(labels) == 0:
                            continue
                        model_name = MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model
                        setting_name = SETTING_MAPPING[setting]
                        sub_dict = {
                            "modality": "Structured EHR",
                            "model": model_name,
                            "setting": setting_name,
                        }
                        for sensitive_attribute in SENSITIVE_ATTRIBUTES[dataset]:
                            sensitive_attributes = extract_sensitive_attributes(dataset, sensitive_attribute.lower())
                            fairness_metrics = calculate_fairness_metrics(labels, preds, sensitive_attributes)
                            for key in fairness_metrics:
                                sub_dict[f"{sensitive_attribute}_{key}"] = fairness_metrics[key]
                        sub_df = pd.DataFrame(sub_dict, index=[df_index])
                        fairness_df = pd.concat([fairness_df, sub_df], axis=0)
                        df_index += 1

                for model in STRUCTURED_EHR_MODELS:
                    for setting in ["0shot", "0shot_unit_range", "1shot_unit_range"]:
                        preds, labels = extract_predictions_and_labels("structured_ehr", dataset, task, model, setting)
                        if len(preds) == 0 or len(labels) == 0:
                            continue
                        model_name = MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model
                        setting_name = SETTING_MAPPING[setting]
                        sub_dict = {
                            "modality": "Structured EHR",
                            "model": model_name,
                            "setting": setting_name,
                        }
                        for sensitive_attribute in SENSITIVE_ATTRIBUTES[dataset]:
                            sensitive_attributes = extract_sensitive_attributes(dataset, sensitive_attribute.lower())
                            fairness_metrics = calculate_fairness_metrics(labels, preds, sensitive_attributes)
                            for key in fairness_metrics:
                                sub_dict[f"{sensitive_attribute}_{key}"] = fairness_metrics[key]
                        sub_df = pd.DataFrame(sub_dict, index=[df_index])
                        fairness_df = pd.concat([fairness_df, sub_df], axis=0)
                        df_index += 1

            # 3. 处理 Multimodal 数据
            if dataset == "mimic-iv":
                # Generation setting
                for model in MULTIMODAL_GENERATION_MODELS:
                    preds, labels = extract_predictions_and_labels("multimodal", dataset, task, model, "0shot_unit_range")
                    if len(preds) == 0 or len(labels) == 0:
                        continue
                    model_name = MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model
                    setting_name = SETTING_MAPPING["0shot_unit_range"]
                    sub_dict = {
                        "modality": "Multimodal",
                        "model": model_name,
                        "setting": setting_name,
                    }
                    for sensitive_attribute in SENSITIVE_ATTRIBUTES[dataset]:
                        sensitive_attributes = extract_sensitive_attributes(dataset, sensitive_attribute.lower())
                        fairness_metrics = calculate_fairness_metrics(labels, preds, sensitive_attributes)
                        for key in fairness_metrics:
                            sub_dict[f"{sensitive_attribute}_{key}"] = fairness_metrics[key]
                    sub_df = pd.DataFrame(sub_dict, index=[df_index])
                    fairness_df = pd.concat([fairness_df, sub_df], axis=0)
                    df_index += 1

                # Tuning setting
                if (dataset, task) in MULTIMODAL_TUNING_CONFIG:
                    model = MULTIMODAL_TUNING_CONFIG[(dataset, task)]
                    for setting in ["add", "concat", "attention", "cross_attention"]:
                        preds, labels = extract_predictions_and_labels("multimodal", dataset, task, model, setting)
                        if len(preds) == 0 or len(labels) == 0:
                            continue
                        model_name = MODEL_NAME_MAPPING[model] if model in MODEL_NAME_MAPPING else model
                        setting_name = SETTING_MAPPING[setting]
                        sub_dict = {
                            "modality": "Multimodal",
                            "model": model_name,
                            "setting": setting_name,
                        }
                        for sensitive_attribute in SENSITIVE_ATTRIBUTES[dataset]:
                            sensitive_attributes = extract_sensitive_attributes(dataset, sensitive_attribute.lower())
                            fairness_metrics = calculate_fairness_metrics(labels, preds, sensitive_attributes)
                            for key in fairness_metrics:
                                sub_dict[f"{sensitive_attribute}_{key}"] = fairness_metrics[key]
                        sub_df = pd.DataFrame(sub_dict, index=[df_index])
                        fairness_df = pd.concat([fairness_df, sub_df], axis=0)
                        df_index += 1

            os.makedirs(f"logs/fairness", exist_ok=True)
            fairness_df.to_csv(f"logs/fairness/fairness_metrics_{dataset}_{task}.csv", index=False)
            generate_latex_from_csv(fairness_df, f"logs/fairness/fairness_metrics_{dataset}_{task}.txt")

if __name__ == "__main__":
    main()