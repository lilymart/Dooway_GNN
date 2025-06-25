import os
import sys
import torch
import pandas as pd
from torch_geometric.nn import to_hetero
import torch.nn as nn
import time

from src.data_loading import load_heterodata
from src.models.GAT import GAT
from src.trainer_ES import train_node_classifier, eval_node_classifier
from src.utils import compute_weights, get_device, set_random_seed



#"$dataset_name" "$mode" "$seed_index" "$seed" "$embeddings_dir" "$models_dir" "$results_dir" "$losses_dir"
if __name__ == "__main__":
    dataset_name = sys.argv[1]
    feats_dim = sys.argv[2]
    target_type = sys.argv[3]
    num_layers = sys.argv[4]
    run = int(sys.argv[5])  # seed_index
    seed = int(sys.argv[6])
    embeddings_dir = sys.argv[7]
    models_dir = sys.argv[8]
    results_dir = sys.argv[9]

    set_random_seed(seed)

    # LOAD THE DATASET
    data = load_heterodata(dataset_name=dataset_name, subset=True, feats_dim=feats_dim, seed=seed)

    # SET THE MODEL
    # model = None
    model = GAT(hidden_channels=64, out_channels=num_classes, dropout=0.3, num_layers=num_layers)
    model = to_hetero(model, data.metadata(), aggr='sum')

    device = torch.device(get_device() if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)  # lr=0.005
    targets = data[target_type].y
    weights = compute_weights(targets).float().to(device)
    criterion = nn.CrossEntropyLoss(weights)

    # TRAIN THE MODEL
    start_time = time.time()
    model = train_node_classifier(model, data, optimizer, criterion, seed, target_type, embeddings_dir, n_epochs=1000, patience=100, epsilon=1e-6)
    end_time = time.time()

    training_time = end_time - start_time
    print(f'Training time: {training_time} seconds')

    # EVALUATE THE MODEL
    f1_micro, f1_macro, f1_weigh, auc, prec, rec = eval_node_classifier(model, data, target_type, seed, embeddings_dir)

    print(f'f1-micro: {f1_micro:.3f}, f1-macro: {f1_macro:.3f}, roc-auc: {auc:.3f}')
    #print(f'precision {prec:.3f}, recall: {rec:.3f})

    # SAVE THE MODEL
    model_path = os.path.join(models_dir, f"{dataset_name}_seed{seed}_model.pth")
    torch.save(model.state_dict(), model_path)

    # SAVE THE RESULTS
    df = pd.DataFrame([{
        'Seed': seed, 'F1_micro': f1_micro, 'F1_macro': f1_macro, 'ROC-AUC': auc, 'Prec': prec, 'Rec': rec, 'Time': training_time
    }])
    results_path = os.path.join(results_dir, f'{dataset_name}_seed{seed}_results.xlsx')
    print(df)
    df.to_excel(results_path, index=False)
    print(f"Saved at {os.path.abspath(results_path)}")

    #if run == 4:
    #merging_results_mode(results_dir, mode)

