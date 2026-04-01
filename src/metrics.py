"""
Evaluation metrics computation
"""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred):
    """Compute comprehensive classification metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = (y_true == y_pred).mean()

    cm = confusion_matrix(y_true, y_pred)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'true_positives': cm[1, 1] if cm.shape == (2, 2) else 0,
        'false_positives': cm[0, 1] if cm.shape == (2, 2) else 0,
        'true_negatives': cm[0, 0] if cm.shape == (2, 2) else 0,
        'false_negatives': cm[1, 0] if cm.shape == (2, 2) else 0,
        'n_samples': len(y_true)
    }


def compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    probs = np.array(probs)
    labels = np.array(labels)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            bin_acc = labels[bin_mask].mean()
            bin_conf = probs[bin_mask].mean()
            bin_size = bin_mask.sum() / len(probs)
            ece += np.abs(bin_acc - bin_conf) * bin_size

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_mask.sum())

    return ece, bin_accuracies, bin_confidences, bin_counts


def brier_score(probs, labels):
    """Compute Brier score."""
    return np.mean((probs - labels) ** 2)
