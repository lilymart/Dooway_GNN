from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import glob
import re
from statistics import stdev
import scipy
from fontTools.ttx import process
from torch import nn


training_seed = [42, 123, 12345, 123123, 2025]

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_base_dir():
    return '/home/martirano/data/dooway'


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def ensure_clean_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Remove the directory and all its contents
    os.makedirs(directory)


def load_from_pickle(pckl_file):
    file = open(pckl_file, 'rb')
    return pickle.load(file)


def save_to_pickle(data_dict, pckl_file):
    with open(pckl_file, 'wb') as file:
        pickle.dump(data_dict, file)


def get_sparse_eye(size):
    eye = scipy.sparse.eye(size)
    coo = eye.tocoo()
    values = coo.data
    indices = torch.LongTensor([coo.row, coo.col])
    i = torch.sparse.FloatTensor(indices, torch.FloatTensor(values), torch.Size([size, size]))
    return i


def compute_weights(targets):
    total_samples = len(targets)
    class_counts = torch.bincount(targets) # Count the occurrences of each class (assuming classes are labeled as 0, 1, 2, ..., n-1)
    weights = total_samples / class_counts.float() # Compute weights inversely proportional to the class frequency
    weights /= weights.sum() # Normalize the weights so they sum to 1

    for i, count in enumerate(class_counts):
        print(f"Number of class {i}s: {count.item()}")
    print(f"Computed weights: {weights}")

    return weights


def compute_weights_safe(targets, num_classes=None):
    total_samples = len(targets)
    class_counts = torch.bincount(targets, minlength=num_classes)

    # Avoid division by zero: set zero counts to 1 (dummy value)
    safe_counts = class_counts.clone()
    safe_counts[safe_counts == 0] = 1

    weights = total_samples / safe_counts.float()
    weights[class_counts == 0] = 0  # set weight=0 for missing classes

    weights /= weights.sum()  # normalize (optional)

    for i, count in enumerate(class_counts):
        print(f"Number of class {i}s: {count.item()}")
    print(f"Computed weights: {weights}")

    return weights


"""
Compute per-class positive weights for multilabel classification.

Parameters: targets : torch.Tensor. Binary label matrix of shape (num_samples, num_classes), where targets[i, c] ∈ {0,1}.
Returns: torch.Tensor. pos_weight vector of shape (num_classes,), suitable for nn.BCEWithLogitsLoss(pos_weight=...).
"""
def compute_pos_weights_multilabel(targets: torch.Tensor) -> torch.Tensor:

    # Count positives and negatives per class
    pos_counts = targets.sum(dim=0)
    neg_counts = targets.size(0) - pos_counts

    # Avoid division by zero (if a class has no positives)
    safe_pos_counts = pos_counts.clone()
    safe_pos_counts[safe_pos_counts == 0] = 1

    # Compute pos_weight = neg / pos
    pos_weight = neg_counts / safe_pos_counts

    # Optional: clip to avoid exploding gradients for extremely rare labels
    pos_weight = torch.clamp(pos_weight, max=50.0)

    # Normalization
    pos_weight = pos_weight / pos_weight.mean()

    # Log some info
    for i, (p, n, w) in enumerate(zip(pos_counts, neg_counts, pos_weight)):
        print(f"Class {i}: pos={int(p)}, neg={int(n)}, pos_weight={float(w):.3f}")

    return pos_weight





"""
Update the label tensor `y` with ground truth from a CSV file. replace rows at IDs with CSV labels.

Args:
    y (torch.Tensor): Original labels tensor [num_nodes, num_classes].
    df: DataFrame with IDs + one-hot labels.
    
Returns:
    updated_y (torch.Tensor): New label tensor.
    ids (np.ndarray): Array of IDs from the CSV.
"""
def update_labels_from_df(y, df_path):

    df = pd.read_parquet(df_path)
    # first column = ids, rest = one-hot labels
    ids = df.iloc[:, 0].values
    y_new = torch.tensor(df.iloc[:, 1:].values, dtype=y.dtype)
    updated_y = y.clone()
    updated_y[ids] = y_new
    return updated_y, ids


"""
Compute accuracy for multi-label classification:
the proportion of samples where at least k labels are correctly predicted.
Args:
    y_true (ndarray): binary matrix of shape (n_samples, n_classes)
    y_pred (ndarray): binary matrix of shape (n_samples, n_classes)
    k (int): number of correct labels required

Returns: float: accuracy score
"""
def at_least_k_accuracy(y_true, y_pred, k=1):

    # Count number of correctly predicted labels per sample
    correct_per_sample = np.sum((y_true == 1) & (y_pred == 1), axis=1)

    # Check if at least k are correct
    success = correct_per_sample >= k
    return np.mean(success)


"""
Format a classification report dictionary into a readable string.
Args:
    report_dict (dict): classification report dictionary
    params_str (str): description of parameters used in the experiment

Returns:
    str: formatted report
"""
def format_classification_report(report_dict, params_str):

    lines = []
    lines.append(f"Parameters: {params_str}\n")

    # Per-class metrics
    for key, metrics in report_dict.items():
        if key.isdigit():  # only class IDs
            lines.append(f"Class {key}:")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1-score:  {metrics['f1-score']:.4f}")
            lines.append(f"  Support:   {int(metrics['support'])}\n")

    # Averages
    lines.append("--- Averages ---")
    for avg_key in ["micro avg", "macro avg", "weighted avg", "samples avg"]:
        if avg_key in report_dict:
            metrics = report_dict[avg_key]
            lines.append(f"{avg_key.title()}:")
            lines.append(f"  Precision: {metrics['precision']:.4f}")
            lines.append(f"  Recall:    {metrics['recall']:.4f}")
            lines.append(f"  F1-score:  {metrics['f1-score']:.4f}")
            lines.append(f"  Support:   {int(metrics['support'])}\n")

    # Other metrics
    lines.append("--- Other metrics ---")
    for metric in ["hamming_loss", "subset_accuracy"]:
        if metric in report_dict:
            lines.append(f"{metric.replace('_',' ').title()}: {report_dict[metric]:.4f}")

    for k in range(1,6):
        key = f"at_least_{k}_label"
        if key in report_dict:
            lines.append(f"At least {k} label(s) correct: {report_dict[key]:.4f}")

    lines.append("\n" + "="*60 + "\n")
    return "\n".join(lines)


"""
Save a classification report to a txt file (appends at the end).
"""
def save_classification_report(report_dict, params_str, file_path="results.txt"):

    report_str = format_classification_report(report_dict, params_str)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(report_str + "\n")
    return report_str  # return so you can also print it


"""
Reads a results txt file containing multiple seed blocks,
extracts relevant metrics, computes mean ± std across seeds,
and returns a LaTeX table row string.
"""
def results_to_latex_row(filepath: str, label: str = "Nostra soluzione", filter_keywords: list[str] = None) -> str:

    with open(filepath, "r") as f:
        text = f.read()

    # Split into separate experiment blocks
    blocks = [b for b in text.strip().split("============================================================") if b.strip()]

    # Filter blocks if requested
    if filter_keywords:
        filtered_blocks = []
        for block in blocks:
            param_line = next((line for line in block.splitlines() if line.startswith("Parameters:")), "")
            if any(keyword in param_line for keyword in filter_keywords):
                filtered_blocks.append(block)
        blocks = filtered_blocks

    if not blocks:
        raise ValueError("No matching blocks found for the given filter keywords.")

    # Regex patterns for each metric
    patterns = {
        "at_least_1": r"At least 1 label\(s\) correct:\s*([\d\.]+)",
        "at_least_3": r"At least 3 label\(s\) correct:\s*([\d\.]+)",
        "at_least_5": r"At least 5 label\(s\) correct:\s*([\d\.]+)",
        "subset_acc": r"Subset Accuracy:\s*([\d\.]+)",
        "hamming_loss": r"Hamming Loss:\s*([\d\.]+)",
        "weighted_prec": r"Weighted Avg:\s*[\s\S]*?Precision:\s*([\d\.]+)",
        "weighted_rec": r"Weighted Avg:\s*[\s\S]*?Recall:\s*([\d\.]+)",
        "weighted_f1": r"Weighted Avg:\s*[\s\S]*?F1-score:\s*([\d\.]+)",
    }

    # Collect values
    results = {key: [] for key in patterns}

    for block in blocks:
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                results[key].append(float(match.group(1)))

    # Compute means and stds
    means = {k: np.mean(v) if v else float("nan") for k, v in results.items()}
    stds = {k: np.std(v) if v else float("nan") for k, v in results.items()}

    # Order for the LaTeX row
    order = [
        "at_least_1",
        "at_least_3",
        "at_least_5",
        "subset_acc",
        "hamming_loss",
        "weighted_prec",
        "weighted_rec",
        "weighted_f1",
    ]

    # Format the LaTeX line
    values = [f"{means[k]:.3f} $\\pm$ {stds[k]:.3f}" for k in order]
    latex_row = f"\\textbf{{{label}}} & " + " & ".join(values) + " \\\\"

    return latex_row

def class_results_to_latex_table(
    filepath: str,
    label_prefix: str = "Noi",
    filter_keywords: list[str] = None
) -> str:
    """
    Reads a results txt file with multiple experiment blocks (different seeds),
    extracts per-class metrics, groups by (Hidden channels, Learning rate, Dropout),
    computes mean ± std across seeds, and returns a LaTeX table string.

    Columns:
      Model | Class | Precision | Recall | F1-score | Support (min) | Support (max)

    - Model is multi-row per configuration.
    - Support uses min and max across seeds, not mean/std.
    """

    with open(filepath, "r") as f:
        text = f.read()

    # Split blocks by separator
    blocks = [b for b in text.strip().split("============================================================") if b.strip()]

    # Optional filtering by parameters
    if filter_keywords:
        filtered_blocks = []
        for block in blocks:
            param_line = next((line for line in block.splitlines() if line.startswith("Parameters:")), "")
            if any(keyword in param_line for keyword in filter_keywords):
                filtered_blocks.append(block)
        blocks = filtered_blocks

    if not blocks:
        raise ValueError("No matching blocks found for the given filter keywords.")

    # Regex patterns
    param_pattern = re.compile(
        r"Hidden channels:(\d+).*?Learning rate:([\d\.]+).*?Dropout:([\d\.]+)"
    )
    class_pattern = re.compile(
        r"Class (\d+):\s*"
        r"Precision:\s*([\d\.]+)\s*"
        r"Recall:\s*([\d\.]+)\s*"
        r"F1-score:\s*([\d\.]+)\s*"
        r"Support:\s*(\d+)",
        re.MULTILINE
    )

    # Group results: model -> class -> metric -> list
    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for block in blocks:
        # Extract hyperparameters
        param_match = param_pattern.search(block)
        if not param_match:
            continue
        hidden, lr, dropout = param_match.groups()
        lr_short = lr.replace(".", "")[1:]  # e.g. 0.0005 -> 0005
        model_tag = f"{label_prefix} {hidden} {lr_short} {dropout}"

        # Extract class metrics
        for match in class_pattern.finditer(block):
            cls = int(match.group(1))
            prec, rec, f1, sup = map(float, match.groups()[1:])
            grouped_results[model_tag][cls]["precision"].append(prec)
            grouped_results[model_tag][cls]["recall"].append(rec)
            grouped_results[model_tag][cls]["f1"].append(f1)
            grouped_results[model_tag][cls]["support"].append(sup)

    # Build table rows
    rows = []
    for model_tag, class_data in grouped_results.items():
        for cls, metrics in sorted(class_data.items()):
            def mean_std(values):
                return f"{np.mean(values):.3f} $\\pm$ {np.std(values):.3f}" if values else "—"

            if metrics["support"]:
                support_min = int(np.min(metrics["support"]))
                support_max = int(np.max(metrics["support"]))
            else:
                support_min = support_max = "—"

            rows.append([
                model_tag,
                cls,
                mean_std(metrics["precision"]),
                mean_std(metrics["recall"]),
                mean_std(metrics["f1"]),
                support_min,
                support_max
            ])

    # Create DataFrame
    df = pd.DataFrame(
        rows,
        columns=["Model", "Class", "Precision", "Recall", "F1-score", "Min Supp", "Max Supp"]
    )
    df = df.sort_values(by=["Model", "Class"]).reset_index(drop=True)

    # === Generate LaTeX manually (for multirow) ===
    latex_lines = []
    latex_lines.append("\\begin{table}[ht]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Per-class results grouped by model configuration}")
    latex_lines.append("\\label{tab:per_class_results}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("Model & Class & Precision & Recall & F1-score & Min Supp & Max Supp\\\\")
    latex_lines.append("\\midrule")

    for model_tag, group_df in df.groupby("Model"):
        n_rows = len(group_df)
        first_row = True
        for _, row in group_df.iterrows():
            if first_row:
                model_cell = f"\\multirow{{{n_rows}}}{{*}}{{{model_tag}}}"
                first_row = False
            else:
                model_cell = ""
            latex_lines.append(
                f"{model_cell} & {int(row['Class'])} & {row['Precision']} & {row['Recall']} & "
                f"{row['F1-score']} & {row['Min Supp']} & {row['Max Supp']} \\\\"
            )
        latex_lines.append("\\midrule")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)



if __name__ == "__main__":
    filepath = os.path.join(get_base_dir(), "Basilicata", "subset", "results", "experiments_results.txt")
    #res = results_to_latex_row(filepath, filter_keywords=["Num layers:3, Hidden channels:64, Learning rate:0.005, Dropout:0.3"])
    res = class_results_to_latex_table(filepath, label_prefix="Noi", filter_keywords=["Num layers:3, Hidden channels:64, Learning rate:0.001, Dropout:0.3"])
    print(res)
