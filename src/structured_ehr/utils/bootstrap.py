import numpy as np

from .metrics import get_all_metrics


def bootstrap(preds, labels, n_iterations=100, seed=42):
    """
    生成自助法重采样的索引。

    参数:
        preds (array-like): 模型预测值。
        labels (array-like): 真实标签。
        n_iterations (int): 自助采样的次数。
        seed (int): 随机种子，用于可复现性。

    返回:
        list of tuples: 每个元组包含一个自助样本的 (preds, labels)。
    """
    assert len(preds) == len(labels), "Predictions and labels must have the same length"
    length = len(preds)

    # 确保是 numpy 数组
    preds = np.array(preds) if not isinstance(preds, np.ndarray) else preds
    labels = np.array(labels) if not isinstance(labels, np.ndarray) else labels

    # 设置随机种子
    rng = np.random.RandomState(seed)

    bootstrapped_samples = []
    for _ in range(n_iterations):
        sample_indices = rng.choice(length, size=length, replace=True)

        sample_preds = preds[sample_indices]
        sample_labels = labels[sample_indices]

        bootstrapped_samples.append((sample_preds, sample_labels))

    return bootstrapped_samples


def calculate_metrics_with_ci(original_preds, original_labels, bootstrapped_samples, config, confidence_level=0.95):
    """
    计算点估计和置信区间。

    参数:
        original_preds (array-like): 原始的完整预测值。
        original_labels (array-like): 原始的完整标签。
        bootstrapped_samples (list): bootstrap函数生成的样本列表。
        config (dict): 配置字典。
        confidence_level (float): 置信水平 (例如 0.95 表示 95% CI)。

    返回:
        dict: 包含每个指标的点估计和置信区间的字典。
              格式: {"metric_name": {"point_estimate": value, "ci": (lower, upper)}}
    """
    # 1. 在原始完整数据集上计算点估计 (Point Estimate)
    point_estimates = get_all_metrics(original_preds, original_labels, config['task'], config['los_info'])

    # 2. 在每个 bootstrap 样本上计算指标，并收集结果
    bootstrap_scores = {key: [] for key in point_estimates.keys()}

    for sample_preds, sample_labels in bootstrapped_samples:
        # 重要：检查自助样本是否至少包含两个类别，否则无法计算AUROC/AUPRC
        if config['task'] != 'los' and len(np.unique(sample_labels)) < 2:
            continue # 跳过这个无法计算的样本

        res = get_all_metrics(sample_preds, sample_labels, config['task'], config['los_info'])
        for k, v in res.items():
            bootstrap_scores[k].append(v)

    # 3. 计算每个指标的置信区间
    final_metrics = {}
    alpha = (1.0 - confidence_level) / 2.0  # 计算分位数的 alpha 值

    for k, v_list in bootstrap_scores.items():
        if not v_list: # 如果列表为空 (例如所有样本都只有一个类别)
            lower, upper = np.nan, np.nan
        else:
            # 使用 np.percentile 计算置信区间的上下限
            lower = np.percentile(v_list, alpha * 100)
            upper = np.percentile(v_list, (1.0 - alpha) * 100)

        final_metrics[k] = {
            "mean": np.mean(v_list),
            "std": np.std(v_list),
            "ci": (lower, upper)
        }

    return final_metrics


def run_bootstrap(preds, labels, config, n_iterations=100, confidence_level=0.95, seed=42):
    """
    执行完整的 bootstrap 流程并返回带有置信区间的结果。
    """
    # 1. 生成 bootstrap 样本
    bootstrap_samples = bootstrap(preds, labels, n_iterations=n_iterations, seed=seed)

    # 2. 计算点估计和置信区间
    metrics = calculate_metrics_with_ci(preds, labels, bootstrap_samples, config, confidence_level)

    return metrics