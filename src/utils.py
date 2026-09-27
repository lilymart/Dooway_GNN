from collections import defaultdict

import torch
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import json
import glob
import re
import math
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
    #return '/home/martirano/data/dooway'
    return "/home/jovyan/data/dooway"


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


def compute_class_weights_multilabel(targets):
    pos_counts = targets.sum(dim=0)
    neg_counts = targets.size(0) - pos_counts

    safe_pos_counts = pos_counts.clone()
    safe_pos_counts[safe_pos_counts == 0] = 1

    class_weight = neg_counts / safe_pos_counts
    class_weight = torch.clamp(class_weight, max=50.0)

    # This normalization makes sense for class-wise weights:
    # it keeps the average loss scale approximately unchanged.
    class_weight = class_weight / class_weight.mean()

    return class_weight





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



###### NEW PER ALBERTO

"""
Saves one classification report (dictionary) as a TXT file.
File name format: report_seed_<seed>.txt
"""
def save_report_to_txt(report_dict, seed, parameter_name, parameter_value, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"report_seed_{seed}_{parameter_name}{parameter_value}.txt")

    with open(filepath, "w") as f:
        for key, value in report_dict.items():
            f.write(f"{key}: {value}\n")

"""
Appends one classification report to the master JSONL file.
Each line is: {"seed": seed, "robustness":robustness, "report": report_dict}
"""
def append_report_to_master(report_dict, seed, parameter_name, parameter_value, output_dir, master_file_name="all_reports.jsonl"):

    entry = {"seed": seed, parameter_name:parameter_value, "report": report_dict}
    master_file = os.path.join(output_dir, master_file_name)
    with open(master_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

def process_master_file(output_dir, master_file_name, out_excel_k, out_excel_metrics):

    master_file = os.path.join(output_dir, master_file_name)
    entries = []

    with open(master_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            entries.append(entry)

    # --------------------------
    # 2. Flatten reports + keep metadata
    # --------------------------
    flattened_reports = []

    for entry in entries:
        rep = entry["report"]
        flat = {
            "seed": entry["seed"],
            "robustness": entry["robustness"]
        }

        for key, value in rep.items():

            # aggregated metrics
            if isinstance(key, str) and key.endswith("avg"):
                # Example: "micro avg": {"precision":..., "recall":..., "f1-score":...}
                name = key.replace(" avg", "")  # "micro", "macro", "weighted", "samples"
                flat[f"precision_{name}"] = value.get("precision", None)
                flat[f"recall_{name}"] = value.get("recall", None)
                flat[f"f1_{name}"] = value.get("f1-score", None)

            # ---- general metrics
            elif key in ["hamming_loss", "subset_accuracy"]:
                flat[key] = value

            # ---- "at_least_k_labels"
            elif isinstance(key, str) and key.startswith("at_least_"):
                flat[key] = value

        flattened_reports.append(flat)

    # --------------------------
    # 3. Convert to DataFrame
    # --------------------------
    df = pd.DataFrame(flattened_reports)

    # ------------------------------------------
    # 4. Build TABLE 1: at_least_k_labels metrics grouped by robustness
    # ------------------------------------------
    k_cols = [c for c in df.columns if c.startswith("at_least_")]

    table_k = []

    for robustness_value, group in df.groupby("robustness"):
        for col in k_cols:
            # Extract k from "at_least_{k}_labels"
            k = int(col.split("_")[2])

            values = group[col].astype(float).values
            table_k.append({
                "model": "our model",
                "robustness": robustness_value,
                "k": k,
                "mean": np.mean(values),
                "std": np.std(values, ddof=1)
            })

    table_k_df = pd.DataFrame(table_k).sort_values(["robustness", "k"])
    table_k_df.to_excel(os.path.join(output_dir, out_excel_k), index=False)

    # ---------------------------------------------------------
    # 5. Build TABLE 2: general metrics + micro/macro/weighted grouped by robustness
    # ---------------------------------------------------------

    metric_groups = [
        "hamming_loss",
        "subset_accuracy",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_micro", "recall_micro", "f1_micro",
        "precision_macro", "recall_macro", "f1_macro",
    ]

    rows = []

    for robustness_value, group in df.groupby("robustness"):

        row = {
            "model": "our model",
            "robustness": robustness_value
        }

        for col in metric_groups:
            if col in group.columns:
                values = group[col].astype(float).values
                row[f"{col}_mean"] = np.mean(values)
                row[f"{col}_std"] = np.std(values, ddof=1)
            else:
                # If column missing → fill with NaN
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan

        rows.append(row)

    table_metrics_df = pd.DataFrame(rows).sort_values("robustness")
    table_metrics_df.to_excel(os.path.join(output_dir, out_excel_metrics), index=False)


def process_master_file_old(output_dir, master_file_name, out_excel_k, out_excel_metrics):

    master_file = os.path.join(output_dir, master_file_name)
    seeds = []
    reports = []

    with open(master_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            seeds.append(entry["seed"])
            reports.append(entry["report"])

    # --------------------------
    # 2. Flatten each report dict
    # --------------------------
    flattened_reports = []

    for rep in reports:
        flat = {}

        # ---- Extract per-class entries (0,1,2,...)
        # ignore integer keys; we keep only aggregated metrics (micro/macro/weighted/samples)
        # but do not crash
        for key, value in rep.items():
            if isinstance(key, str) and key.endswith("avg"):
                # Example: "micro avg": {"precision":..., "recall":..., "f1-score":...}
                name = key.replace(" avg", "")  # "micro", "macro", "weighted", "samples"
                flat[f"precision_{name}"] = value.get("precision", None)
                flat[f"recall_{name}"] = value.get("recall", None)
                flat[f"f1_{name}"] = value.get("f1-score", None)

            # ---- Extract general metrics
            elif key in ["hamming_loss", "subset_accuracy"]:
                flat[key] = value

            # ---- Extract "at_least_k_labels"
            elif isinstance(key, str) and key.startswith("at_least_"):
                flat[key] = value

        flattened_reports.append(flat)

    # --------------------------
    # 3. Convert to DataFrame
    # --------------------------
    df = pd.DataFrame(flattened_reports)
    df["seed"] = seeds

    # ------------------------------------------
    # 4. Build TABLE 1: at_least_k_labels metrics
    # ------------------------------------------
    k_cols = [c for c in df.columns if c.startswith("at_least_")]

    table_k = []
    for col in k_cols:
        # Extract k from "at_least_{k}_labels"
        k = int(col.split("_")[2])

        values = df[col].astype(float).values
        table_k.append({
            "model": "our model",
            "k": k,
            "mean": np.mean(values),
            "std": np.std(values, ddof=1)
        })

    table_k_df = pd.DataFrame(table_k).sort_values("k")
    table_k_df.to_excel(os.path.join(output_dir, out_excel_k), index=False)

    # ---------------------------------------------------------
    # 5. Build TABLE 2: general metrics + micro/macro/weighted
    # ---------------------------------------------------------

    metric_groups = [
        "hamming_loss",
        "subset_accuracy",
        "precision_weighted", "recall_weighted", "f1_weighted",
        "precision_micro", "recall_micro", "f1_micro",
        "precision_macro", "recall_macro", "f1_macro",
    ]

    row = {"model": "our model"}

    for col in metric_groups:
        if col in df.columns:
            values = df[col].astype(float).values
            row[f"{col}_mean"] = np.mean(values)
            row[f"{col}_std"] = np.std(values, ddof=1)
        else:
            # If column missing → fill with NaN
            row[f"{col}_mean"] = np.nan
            row[f"{col}_std"] = np.nan

    table_metrics_df = pd.DataFrame([row])
    table_metrics_df.to_excel(os.path.join(output_dir, out_excel_metrics), index=False)


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
    return report_str


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



# NON COMMENTIAMO... X ALBERTO

def normalize_metric_name(name):
    """Normalize metric names by collapsing whitespace."""
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def classify_metric(name):
    """Classify real metric names into logical keys."""
    n = normalize_metric_name(name).lower()

    if "hamming_loss" in n:
        return "HAMMING"

    if "f1-score" in n:
        if "micro" in n:
            return "F1_MICRO"
        if "macro" in n:
            return "F1_MACRO"
        if "weighted" in n:
            return "F1_WEIGHTED"

    if "precision" in n and "weighted" in n:
        return "PREC_W"

    if "recall" in n and "weighted" in n:
        return "RECALL_W"

    if "precision" in n and "macro" in n:
        return "PREC_MACRO"
    if "recall" in n and "macro" in n:
        return "RECALL_MACRO"

    return None


def parse_results_file(filepath):
    models = {}
    current_model = None

    # Example: === Aggregati su 5 run: Gradient Boosting ===
    header_pattern = re.compile(r"Aggregati su \d+ run:\s*(.+?)\s*===")

    # Example:
    # 2025-11-18 ... | INFO | F1-SCORE  (micro): mean=0.8213 var=0.000024
    metric_pattern = re.compile(
        r"\|\s*([A-Za-z0-9_\-\s\(\)]+):\s*mean=([0-9.]+)\s*var=([0-9.]+)"
    )
    with open(filepath, "r") as f:
        for line in f:

            # Header
            header = header_pattern.search(line)
            if header:
                current_model = header.group(1).strip()
                models[current_model] = {}
                continue

            # Metric line: extract after last pipe |
            metric = metric_pattern.search(line)
            if metric and current_model:
                raw_name = metric.group(1).strip()
                mean = float(metric.group(2))
                var = float(metric.group(3))

                key = classify_metric(raw_name)
                if key:
                    models[current_model][key] = (mean, var)

    return models


def fmt(mean, var):
    std = math.sqrt(var)
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def build_tables(filepath, table_type=1):
    data = parse_results_file(filepath)

    if table_type == 1:
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lcccc}",
            "\\hline",
            "Model & Hamming Loss & F1 (micro) & F1 (macro) & F1 (weighted)\\\\",
            "\\hline"
        ]

        for model, m in data.items():
            latex.append(
                f"{model} & "
                f"{fmt(*m['HAMMING'])} & "
                f"{fmt(*m['F1_MICRO'])} & "
                f"{fmt(*m['F1_MACRO'])} & "
                f"{fmt(*m['F1_WEIGHTED'])} \\\\"
            )

        latex += ["\\hline", "\\end{tabular}", "\\end{table}"]
        return "\n".join(latex)

    if table_type == 2:
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Model & Precision (w) & Recall (w) & F1 (w)\\\\",
            "\\hline"
        ]

        for model, m in data.items():
            latex.append(
                f"{model} & "
                f"{fmt(*m['PREC_W'])} & "
                f"{fmt(*m['RECALL_W'])} & "
                f"{fmt(*m['F1_WEIGHTED'])} \\\\"
            )

        latex += ["\\hline", "\\end{tabular}", "\\end{table}"]
        return "\n".join(latex)

    if table_type == 3:
        latex = [
            "\\begin{table}[h]",
            "\\centering",
            "\\begin{tabular}{lccc}",
            "\\hline",
            "Model & Precision (macro) & Recall (macro) & F1 (macro)\\\\",
            "\\hline"
        ]

        for model, m in data.items():
            latex.append(
                f"{model} & "
                f"{fmt(*m['PREC_MACRO'])} & "
                f"{fmt(*m['RECALL_MACRO'])} & "
                f"{fmt(*m['F1_MACRO'])} \\\\"
            )

        latex += ["\\hline", "\\end{tabular}", "\\end{table}"]
        return "\n".join(latex)




def reports_to_excel(infile, outfile):
    common_dir = os.path.join(get_base_dir(), "Basilicata","subset", "results", "ablation_tracker")
    input_file = os.path.join(common_dir, infile)
    output_file = os.path.join(common_dir, outfile)

    rows = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            result = json.loads(line)

            seed = result["seed"]
            use_attrs = result.get("use_attrs")
            report = result["report"]

            row = {
                "seed": seed,
                "use_attrs": use_attrs,

                # Main classification metrics
                "HL": report["hamming_loss"],
                "F1-micro": report["micro avg"]["f1-score"],
                "F1-macro": report["macro avg"]["f1-score"],
                "F1-weighted": report["weighted avg"]["f1-score"],

                # Precision / recall
                "Prec-micro": report["micro avg"]["precision"],
                "Rec-micro": report["micro avg"]["recall"],

                "Prec-macro": report["macro avg"]["precision"],
                "Rec-macro": report["macro avg"]["recall"],

                "Prec-weighted": report["weighted avg"]["precision"],
                "Rec-weighted": report["weighted avg"]["recall"],

                # Other metrics
                "Subset accuracy": report["subset_accuracy"],

                # Training / computational cost
                "Epochs": report.get("epochs_trained"),
                "FLOPs-fwd": report.get("FLOPs_fwd"),
                "FLOPs-bwd": report.get("FLOPs_bwd"),
                "FLOPs-tot": report.get("FLOPs_tot"),
                "BOPs-fwd": report.get("BOPs_fwd"),
                "BOPs-bwd": report.get("BOPs_bwd"),
                "BOPs-tot": report.get("BOPs_tot"),
            }

            rows.append(row)

    df = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # Mean ± standard deviation across seeds
    # ---------------------------------------------------------

    final_row = {
        "seed": "Mean ± std",
        "use_attrs": (
            df["use_attrs"].iloc[0]
            if df["use_attrs"].nunique() == 1
            else ""
        ),
    }

    numeric_columns = [
        col for col in df.columns
        if col not in ["seed", "use_attrs"]
    ]

    for col in numeric_columns:
        values = pd.to_numeric(df[col], errors="coerce")

        mean = values.mean()
        std = values.std(ddof=1)

        # Scientific notation for computational cost
        if "FLOPs" in col or "BOPs" in col:
            final_row[col] = f"{mean:.3e} ± {std:.3e}"

        # Epochs
        elif col == "Epochs":
            final_row[col] = f"{mean:.1f} ± {std:.1f}"

        # Classification metrics
        else:
            final_row[col] = f"{mean:.4f} ± {std:.4f}"

    df_final = pd.concat(
        [df, pd.DataFrame([final_row])],
        ignore_index=True
    )

    # ---------------------------------------------------------
    # Save Excel
    # ---------------------------------------------------------

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_final.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        worksheet = writer.sheets["Summary"]

        # Freeze header
        worksheet.freeze_panes = "A2"

        # Basic column widths
        for column in worksheet.columns:
            max_length = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in column
            )

            worksheet.column_dimensions[
                column[0].column_letter
            ].width = min(max_length + 2, 25)

    print(f"Saved: {output_file}")

    return df_final




if __name__ == "__main__":
    #filepath = os.path.join(get_base_dir(), "Basilicata", "subset", "results", "experiments_results.txt")
    #res = results_to_latex_row(filepath, filter_keywords=["Num layers:3, Hidden channels:64, Learning rate:0.005, Dropout:0.3"])
    #res = class_results_to_latex_table(filepath, label_prefix="Noi", filter_keywords=["Num layers:3, Hidden channels:64, Learning rate:0.001, Dropout:0.3"])
    #print(res)

    """
    suffix = "robustness" #"ablation"
    process_master_file(
        output_dir = "/home/martirano/data/dooway/Basilicata/subset/results",
        master_file_name=f"all_reports_{suffix}.txt", #.jsonl
        out_excel_k=f"table_k_{suffix}.xlsx",
        out_excel_metrics=f"table_metrics_{suffix}.xlsx"
    )
    """
    print("ciao")
    reports_to_excel("all_reports.jsonl","all_reports_summary.xlsx")


    #output_dir = "/home/martirano/data/dooway/Basilicata/subset/results"
    #competitors_results = "Baseline_Evaluation_new_2_20251118_094248.log"
    #file_results = os.path.join(output_dir, competitors_results)
    #table1 = build_tables(file_results, table_type=1)
    #print(table1)

    #table2 = build_tables(file_results, table_type=2)
    #print(table2)

    #table3 = build_tables(file_results, table_type=3)
    #print(table3)


