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

    for k in range(1,4):
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
