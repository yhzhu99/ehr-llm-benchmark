import numpy as np

from structured_ehr.utils.metrics import get_all_metrics


def bootstrap(preds, labels, K=100, N=1000, seed=42):
    assert len(preds) == len(labels), "Predictions and labels must have the same length"
    length = len(preds)

    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Check if preds and labels are numpy arrays, if not convert them to numpy arrays
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)

    # Initialize a list to store bootstrap samples
    bootstrapped_samples = []

    # Create K bootstrap samples
    for _ in range(K):
        # Sample with replacement from the indices
        sample_indices = np.random.choice(length, length, replace=True)

        # Use the sampled indices to get the bootstrap sample of preds and labels
        sample_preds = preds[sample_indices]
        sample_labels = labels[sample_indices]

        # Store the bootstrap samples
        bootstrapped_samples.append((sample_preds, sample_labels))

    return bootstrapped_samples


def export_metrics(bootstrapped_samples, config):
    metrics = {}

    for sample in bootstrapped_samples:
        sample_preds, sample_labels = sample
        res = get_all_metrics(sample_preds, sample_labels, config['task'], None)

        for k, v in res.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    # convert to numpy array
    for k, v in metrics.items():
        metrics[k] = np.array(v)

    # calculate mean and std
    for k, v in metrics.items():
        metrics[k] = {"mean": np.mean(v), "std": np.std(v)}
    return metrics


def run_bootstrap(preds, labels, config, seed=42):
    bootstrap_samples = bootstrap(preds, labels, seed=seed)
    metrics = export_metrics(bootstrap_samples, config)
    return metrics