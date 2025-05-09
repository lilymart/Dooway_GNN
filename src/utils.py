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
    return '/mnt/nas/martirano/'


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





if __name__ == "__main__":
    dataset_name = "politifact"  #"politifact" "mumin"





