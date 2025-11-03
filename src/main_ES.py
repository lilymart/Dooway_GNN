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
from src.trainer_ES import train_node_classifier, eval_node_classifier_multilabel, \
    eval_node_classifier_multiclass
from src.utils import get_device, set_random_seed, compute_weights_safe, get_base_dir, save_classification_report, \
    compute_pos_weights_multilabel

if __name__ == "__main__":

    """
    dataset_name = sys.argv[1] #"Basilicata"
    feats_dim = sys.argv[2] #"256"
    target_type = sys.argv[3] #"user"
    multilabel = sys.argv[4].lower() == "true" #True
    seed = int(sys.argv[5]) #42
    num_layers = int(sys.argv[6]) #3
    hidden_channels = int(sys.argv[7]) #32  # 64
    dropout = float(sys.argv[8]) #0.3
    embeddings_dir = None #sys.argv[8]
    results_dir = sys.argv[9] #os.path.join(get_base_dir(), dataset_name, "subset", "results")
    """

    parser = argparse.ArgumentParser(description="Run GAT experiment")

    parser.add_argument("--dataset_name", type=str, default="Basilicata",
                        help="Name of the dataset (e.g. Basilicata)")
    parser.add_argument("--feats_dim", type=int, default=256,
                        help="Size of node features")
    parser.add_argument("--target_type", type=str, default="user",
                        help="Target type (e.g. user)")
    parser.add_argument("--multilabel", type=lambda x: x.lower() == "true", default=True,
                        help="Whether task is multilabel (true/false)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for training")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of GAT layers")
    parser.add_argument("--hidden_channels", type=int, default=128,
                        help="Size oh hidden channels")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--learning_rate", type=float, default=0.005,
                        help="Learning rate")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.getcwd(), "results"),
                        help="Path (directory) to store results")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    feats_dim = args.feats_dim
    target_type = args.target_type
    multilabel = args.multilabel
    seed = args.seed
    num_layers = args.num_layers
    hidden_channels = args.hidden_channels
    dropout = args.dropout
    learning_rate = args.learning_rate
    results_dir = args.results_dir
    embeddings_dir = None

    set_random_seed(seed)

    # LOAD THE DATASET
    fixed_test_set = False
    experts_test_set = False
    max_k = 8
    data = load_heterodata(dataset_name=dataset_name, subset=True, feats_dim=feats_dim, multilabel=multilabel, max_k= max_k, fixed_test_set=fixed_test_set, experts_test_set=experts_test_set, seed=seed)

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

    params_str = f"Num layers:{num_layers}, Hidden channels:{hidden_channels}, Dropout:{dropout}, max k:{max_k}, fixed test set:{fixed_test_set}, experts test set:{experts_test_set}, Seed:{seed}."
    report_str = save_classification_report(report_dict, params_str, os.path.join(results_dir, "experiments_results.txt"))
    print(report_str)

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




