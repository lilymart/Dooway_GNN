import os
import sys
import torch
import pandas as pd
import numpy as np
from torch_geometric.nn import to_hetero
import torch.nn as nn
import time

from src.data_loading import load_heterodata
from src.models.GAT import My_GAT
from src.models.HeteroGAT_unsup import HeteroGAT
from src.trainer_ES import train_node_classifier, eval_node_classifier, eval_node_classifier_multilabel, \
    eval_node_classifier_multiclass
from src.utils import compute_weights, get_device, set_random_seed, compute_weights_safe

#"$dataset_name" "$mode" "$seed_index" "$seed" "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"
if __name__ == "__main__":

    dataset_name = "Basilicata" #sys.argv[1]
    feats_dim = "256" #sys.argv[2]
    target_type = "user" #sys.argv[3]
    multilabel = True
    num_layers = 2 #sys.argv[4]
    run = 0 #int(sys.argv[5])  # seed_index
    seed = 42 #int(sys.argv[6])
    embeddings_dir = None #sys.argv[7]
    #models_dir = sys.argv[8]
    results_dir = sys.argv[9]

    set_random_seed(seed)

    # LOAD THE DATASET
    data = load_heterodata(dataset_name=dataset_name, subset=True, feats_dim=feats_dim, multilabel=multilabel, seed=seed)

    if multilabel:
        num_classes = data[target_type].y.size(1)
    else:
        num_classes = int(data[target_type].y.max().item()) + 1

    # Model
    model = My_GAT(hidden_channels=64, out_channels=num_classes, dropout=0.3, num_layers=num_layers)
    model = to_hetero(model, data.metadata(), aggr='sum')

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)  # lr=0.005

    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        targets = data[target_type].y
        weights = compute_weights_safe(targets, num_classes).float().to(device)
        criterion = nn.CrossEntropyLoss(weights)

    # Train
    start_time = time.time()
    model = train_node_classifier(model, data, optimizer, criterion, target_type, multilabel, n_epochs=600, patience=100, epsilon=1e-6)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    # Evaluation
    #f1_micro, f1_macro, f1_weigh, auc, prec, rec = eval_node_classifier(model, data, target_type, num_layers, seed, embeddings_dir, split="test")

    report_path = os.path.join(results_dir, f"{num_layers}layers_seed{seed}_classification_report.xlsx")

    if multilabel:
        f1_micro, f1_macro, f1_weigh, auc, prec, rec = eval_node_classifier_multilabel(
            model, data, target_type, split="test", report_path=report_path
        )
    else:
        f1_micro, f1_macro, f1_weigh, auc, prec, rec = eval_node_classifier_multiclass(
            model, data, target_type, split="test", report_path=report_path
        )

    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}')
    print(f'precision {prec}, recall: {rec}')

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

    """
    # SAVE THE MODEL
    model_path = os.path.join(models_dir, f"{num_layers}layers_seed{seed}_model.pth")
    torch.save(model.state_dict(), model_path)

    # SAVE THE RESULTS
    df = pd.DataFrame([{
        'Seed': seed,
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'ROC-AUC': auc,
        'Prec': prec,
        'Rec': rec,
        'Time': training_time
    }])
    results_path = os.path.join(results_dir, f'{num_layers}layers_seed{seed}_results.xlsx')
    print(df)
    df.to_excel(results_path, index=False)
    print(f"Saved at {os.path.abspath(results_path)}")
    """


