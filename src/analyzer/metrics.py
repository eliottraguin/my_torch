"""Classification metrics shared by training observers and dashboards."""

import numpy as np


def classification_metrics(confusion, labels):
    matrix = np.asarray(confusion, dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)) or (matrix < 0).any():
        raise ValueError('Invalid confusion matrix')
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    precision = np.divide(correct, predicted, out=np.zeros(len(labels)), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(len(labels)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(labels)), where=(precision + recall) != 0)
    total = int(matrix.sum())
    return {
        'count': total, 'correct': int(correct.sum()),
        'accuracy': float(correct.sum() / total) if total else None,
        'macro_f1': float(f1.mean()) if total else None,
        'balanced_accuracy': float(recall[support > 0].mean()) if total else None,
        'confusion': matrix.tolist(),
        'per_class': [{'label': label, 'precision': float(precision[i]),
                       'recall': float(recall[i]), 'f1': float(f1[i]),
                       'support': int(support[i])} for i, label in enumerate(labels)],
    }


def evaluate_metrics(network, examples, *, check_cancel=None):
    """Evaluate a fixed model; no parameter updates or random sampling."""
    if not examples:
        raise ValueError('Cannot evaluate an empty dataset')
    confusion = np.zeros((network.output_size, network.output_size), dtype=np.int64)
    loss = 0.0
    for encoded, target in examples:
        if check_cancel is not None:
            check_cancel()
        output = network.forward_pass(encoded)[-1]
        if not np.isfinite(output).all():
            raise ValueError('Non-finite model output during evaluation')
        loss += 0.5 * float(np.sum((output - target) ** 2))
        confusion[int(np.argmax(target)), int(np.argmax(output))] += 1
    result = classification_metrics(confusion, network.labels)
    result['loss'] = loss / len(examples)
    if not np.isfinite(result['loss']):
        raise ValueError('Non-finite evaluation loss')
    return result
