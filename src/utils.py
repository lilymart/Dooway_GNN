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


"""Extracts useful metrics from sklearn classification_report output."""
def parse_classification_report(report_dict):

    results = {}

    # Per-class metrics
    results["per_class"] = {
        label: {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1-score"],
            "support": metrics["support"],
        }
        for label, metrics in report_dict.items()
        if label not in ("accuracy", "macro avg", "weighted avg", "micro avg")
    }

    # Aggregated metrics
    results["macro avg"] = report_dict.get("macro avg", {})
    results["weighted avg"] = report_dict.get("weighted avg", {})
    if "micro avg" in report_dict:
        results["micro avg"] = report_dict["micro avg"]

    # Accuracy
    if "accuracy" in report_dict:
        results["accuracy"] = report_dict["accuracy"]

    return results






