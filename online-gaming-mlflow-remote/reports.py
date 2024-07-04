import pandas as pd


def make_classification_report_frame(
    report,
    classes,
):
    """
    This function transforms the classification report dictionary into a pandas DataFrame.

    Parameters:
    report (dict): The dictionary resulting from calling `classification_report` with `output_dict=True`.
    classes (list): The list of classes used in the classification task.

    Returns:
    DataFrame: A pandas DataFrame where each row corresponds to a class and each column corresponds to a metric.
               The metrics are 'precision', 'recall', 'f1-score', and 'support'.
               The DataFrame also includes rows for 'micro avg', 'macro avg', and 'weighted avg' if they are present in the report.

    Example:
    >>> report = classification_report(y_true, y_pred, output_dict=True)
    >>> classes = ['class1', 'class2']
    >>> df = make_classification_report_frame(report, classes)
    """
    metrics = [
        "precision",
        "recall",
        "f1-score",
        "support",
    ]
    data = []
    for label in classes:
        data.append([label] + [report[label][metric] for metric in metrics])
    general = [
        "micro avg",
        "macro avg",
        "weighted avg",
    ]
    for label in general:
        if label in report and isinstance(report[label], dict):
            data.append([label] + [report[label].get(metric, None) for metric in metrics])
    return pd.DataFrame(data, columns=["label"] + metrics)
