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
from floppy import FLOPpyTracker

import warnings

from src.models.GAT_enhanced import My_GAT_enhanced

warnings.filterwarnings("ignore")

from src.data_loading import load_heterodata
from src.models.GAT import My_GAT
from src.models.HAN import HAN
from src.models.HGT import HGT
from src.models.SimpleHGN import SimpleHGN
from src.trainer_ES import train_node_classifier, eval_node_classifier_multilabel, \
    eval_node_classifier_multiclass
from src.utils import get_device, set_random_seed, compute_weights_safe, get_base_dir, save_classification_report, \
    compute_pos_weights_multilabel, save_report_to_txt, append_report_to_master, compute_class_weights_multilabel

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
    parser.add_argument("--model_name", type=str, default="LANTERN",
                        help="Name of the model")
    parser.add_argument("--feats_dim", type=int, default=256,
                        help="Size of node features")
    parser.add_argument("--target_type", type=str, default="user",
                        help="Target type (e.g. user)")
    parser.add_argument("--multilabel", type=lambda x: x.lower() == "true", default=True,
                        help="Whether task is multilabel (true/false)")
    parser.add_argument("--use_attrs", type=lambda x: x.lower() == "true", default=True,
                        help="If false, all attributes are masked (text and imgs)")
    parser.add_argument("--use_imgs", type=lambda x: x.lower() == "true", default=True,
                        help="If false, images are masked")
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
    parser.add_argument("--robustness", type=int, default=100,
                        help="Percentage of training set")
    parser.add_argument("--results_dir", type=str, default=os.path.join(os.getcwd(), "results"),
                        help="Path (directory) to store results")

    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = args.model_name
    feats_dim = args.feats_dim
    target_type = args.target_type
    multilabel = args.multilabel
    use_attrs = args.use_attrs
    use_imgs = args.use_imgs
    seed = args.seed
    num_layers = args.num_layers
    hidden_channels = args.hidden_channels
    dropout = args.dropout
    learning_rate = args.learning_rate
    robustness = args.robustness
    results_dir = args.results_dir
    embeddings_dir = None

    set_random_seed(seed)

    # LOAD THE DATASET
    fixed_test_set = False
    experts_test_set = False
    max_k = 8
    data = load_heterodata(dataset_name=dataset_name, subset=True, feats_dim=feats_dim, use_attrs=use_attrs, use_imgs=use_imgs, multilabel=multilabel, max_k= max_k, fixed_test_set=fixed_test_set, experts_test_set=experts_test_set, seed=seed, robustness=robustness)

    if multilabel:
        num_classes = data[target_type].y.size(1)
    else:
        num_classes = int(data[target_type].y.max().item()) + 1

    edge_attr_dims = {}

    for edge_type in data.edge_types:
        store = data[edge_type]
        if "edge_attr" in store:
            edge_attr = store.edge_attr
            edge_attr_dims[edge_type] = 1 if edge_attr.dim() == 1 else edge_attr.size(-1)

    # Model
    #model = My_GAT(hidden_channels=hidden_channels, out_channels=num_classes, dropout=dropout, num_layers=num_layers) #hidden_channels=64
    #model = to_hetero(model, data.metadata(), aggr='sum')
    if model_name == "LANTERN":
        model = My_GAT_enhanced(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers,
            heads=1,
            edge_attr_dims=edge_attr_dims,
        )
    elif model_name == "HAN":
        model = HAN(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers,
            heads=1,
        )
    elif model_name == "HGT_single_head":
        model = HGT(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers,
            heads=1,
        )
    elif model_name == "HGT":
        model = HGT(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers,
            heads=8,
        )
    elif model_name == "SimpleHGN":
        model = SimpleHGN(
            metadata=data.metadata(),
            target_type=target_type,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=dropout,
            num_layers=num_layers,
            heads=1,
            edge_dim=64,
            beta=0.05,
        )
    else: 
        raise ValueError(f"Unknown model_name: {model_name}. Choose among LANTERN, HAN, HGT, SimpleHGN.")

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # lr=0.005

    targets = data[target_type].y

    if multilabel:
        train_mask = data[target_type].train_mask
        train_targets = targets[train_mask]
        class_weight = compute_class_weights_multilabel(train_targets).float().to(device)
        criterion = nn.BCEWithLogitsLoss(weight=class_weight)
    else:
        weights = compute_weights_safe(targets, num_classes).float().to(device)
        criterion = nn.CrossEntropyLoss(weights)

    tracker = FLOPpyTracker(run_name=f"{model_name}_experiment")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Train

    start_time = time.time()
    tracker.run(model=model, optimizer=optimizer, loss_fn=criterion)
    model, training_info = train_node_classifier(model, data, optimizer, criterion, target_type, multilabel, n_epochs=500, patience=100, epsilon=1e-6)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0

    training_info["training_time"] = training_time
    training_info["num_trainable_params"] = num_trainable_params
    training_info["peak_gpu_memory_mb"] = peak_gpu_memory_mb

    report = tracker.report()


    if multilabel:
        report_dict = eval_node_classifier_multilabel(model, data, target_type, report, training_info, split="test")
    else:
        report_dict = eval_node_classifier_multiclass(model, data, target_type, split="test")


    if multilabel:
        model.eval()

        with torch.no_grad():
            out, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

            # Probabilities for all target nodes
            prob = torch.sigmoid(out[target_type])

            # Keep only test nodes
            test_mask = data[target_type].test_mask
            test_prob = prob[test_mask].detach().cpu()
            test_idx = torch.where(test_mask)[0].cpu()

        os.makedirs(results_dir, exist_ok=True)


    f1_macro = report_dict["macro avg"]["f1-score"]
    f1_weighted = report_dict["weighted avg"]["f1-score"]

    hamming = report_dict.get("hamming_loss", None)
    subset_acc = report_dict.get("subset_accuracy", None)

    print(f'f1-weigh: {f1_weighted:.3f}, f1-macro: {f1_macro:.3f}')
    print("ALL METRICS")

    save_report_to_txt(report_dict, seed, "use_attrs", use_attrs, results_dir)
    append_report_to_master(report_dict, seed, "use_attrs", use_attrs, results_dir)

    # Save target embeddings
    model.eval()

    with torch.no_grad():
        out, embeddings = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
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




