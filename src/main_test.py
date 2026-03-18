import os
import argparse
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import hamming_loss
from torch_geometric.nn import to_hetero
import torch.nn as nn
import time

from src.data_loading import load_heterodata
from src.models.GAT import My_GAT
from src.plots import plot_labels_distribution
from src.trainer_ES import train_node_classifier, eval_node_classifier_multilabel, \
    eval_node_classifier_multiclass
from src.utils import get_device, set_random_seed, compute_weights_safe, get_base_dir, save_classification_report, \
    compute_pos_weights_multilabel, save_report_to_txt, append_report_to_master

if __name__ == "__main__":

    dataset_name = "Basilicata"
    feats_dim = 256
    target_type = "user"
    multilabel = True
    seed = 42
    num_layers = 3
    hidden_channels = 64
    dropout = 0.3
    learning_rate = 0.001
    results_dir = os.path.join(get_base_dir(), dataset_name, "subset", "results")
    embeddings_dir = None

    set_random_seed(seed)

    # LOAD THE DATASET
    fixed_test_set = False
    experts_test_set = False
    max_k = 8
    data = load_heterodata(dataset_name=dataset_name, subset=True, feats_dim=feats_dim, multilabel=multilabel, max_k= max_k, fixed_test_set=fixed_test_set, experts_test_set=experts_test_set, seed=seed)

    plot_labels_distribution(data, results_dir)


    if multilabel:
        num_classes = data[target_type].y.size(1)
    else:
        num_classes = int(data[target_type].y.max().item()) + 1



    # Model
    model = My_GAT(hidden_channels=hidden_channels, out_channels=num_classes, dropout=dropout, num_layers=num_layers) #hidden_channels=64
    model = to_hetero(model, data.metadata(), aggr='sum')

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-3)  # lr=0.005

    targets = data[target_type].y

    if multilabel:
        pos_weight = compute_pos_weights_multilabel(targets).float().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight)
    else:
        weights = compute_weights_safe(targets, num_classes).float().to(device)
        criterion = nn.CrossEntropyLoss(weights)

    # Train
    start_time = time.time()
    model = train_node_classifier(model, data, optimizer, criterion, target_type, multilabel, n_epochs=500, patience=100, epsilon=1e-6)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')


    if multilabel:
        report_dict = eval_node_classifier_multilabel(model, data, target_type, split="test")
    else:
        report_dict = eval_node_classifier_multiclass(model, data, target_type, split="test")

    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_weighted = report_dict["weighted avg"]["f1-score"]

    hamming = report_dict.get("hamming_loss", None)
    subset_acc = report_dict.get("subset_accuracy", None)

    print(f'f1-weigh: {f1_weighted:.3f}, f1-macro: {f1_macro:.3f}')
    print("ALL METRICS")

    #params_str = f"Num layers:{num_layers}, Hidden channels:{hidden_channels}, Dropout:{dropout}, max k:{max_k}, fixed test set:{fixed_test_set}, experts test set:{experts_test_set}, Seed:{seed}."
    #report_str = save_classification_report(report_dict, params_str, os.path.join(results_dir, "experiments_results.txt"))
    #print(report_str)
    # NEW PER ALBERTO
    save_report_to_txt(report_dict, seed, results_dir)
    append_report_to_master(report_dict, seed, results_dir)

    # Save target embeddings
    model.eval()

    with torch.no_grad():
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        all_embeddings = embeddings[target_type].cpu().numpy()
        labels = data[target_type].y.cpu().numpy()
        node_ids = np.arange(len(all_embeddings))

    if embeddings_dir is not None:
        os.makedirs(embeddings_dir, exist_ok=True)

        # Save as .npy (fast loading)
        npy_path = os.path.join(embeddings_dir, f"{num_layers}layers_seed{seed}.npy")
        np.save(npy_path, all_embeddings)
        print(f"Saved embeddings (NumPy) to {npy_path}")

        # Save as .csv
        df = pd.DataFrame(all_embeddings)
        df.insert(0, "node_id", node_ids)
        df["label"] = labels
        csv_path = os.path.join(embeddings_dir, f"{num_layers}layers_seed{seed}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved embeddings (CSV) to {csv_path}")




