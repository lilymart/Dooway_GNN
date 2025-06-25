import os
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def train_node_classifier(model, data, optimizer, criterion, seed, target_type, embeddings_dir, n_epochs=200, patience=20, epsilon=1e-4):

    best_val_f1 = 0.0  # To keep track of the best validation F1 score
    best_model_state = None  # To store the best model's state
    epochs_without_improvement = 0  # To track epochs without improvement

    #loss_values = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x_dict, data.edge_index_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()

        pred = out[target_type].argmax(dim=1)  ## Use the class with highest probability.

        f1_micro, f1_macro, f1_weigh, auc, precision_0, recall_0, precision_1, recall_1 = eval_node_classifier(model, data, target_type, seed, embeddings_dir)

        #loss_values.append(loss.item())

        if f1_macro > best_val_f1 + epsilon:
            best_val_f1 = f1_macro
            best_model_state = model.state_dict()  # Save the best model state
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1


        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val f1_macro: {best_val_f1:.3f}")
            break

        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val f1_micro: {f1_micro:.3f}, Val f1_macro: {f1_macro:.3f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    return model


def eval_node_classifier(model, data, target_type, seed, embeddings_dir):
    model.eval()
    with torch.no_grad():
        # pred = model(data.x_dict, data.edge_index_dict)['claim'].argmax(dim=-1)
        out, embeddings = model(data.x_dict, data.edge_index_dict)
        pred = out[target_type].argmax(dim=-1)
        mask = data[target_type].val_mask
        # correct = (pred[mask] == data['claim'].y[mask]).sum()
        f1_micro = f1_score(data[target_type].y.cpu(), pred.cpu(), average='micro')
        f1_macro = f1_score(data[target_type].y.cpu(), pred.cpu(), average='macro')
        f1_weigh = f1_score(data[target_type].y.cpu(), pred.cpu(), average='weighted')
        auc = roc_auc_score(data[target_type].y.cpu(), pred.cpu(), average='weighted')
        prec = precision_score(data[target_type].y.cpu(), pred.cpu(), average=None) #If None, the metrics for each class are returned
        rec = recall_score(data[target_type].y.cpu(), pred.cpu(), average=None)

        # Save embeddings for validation set
        val_embeddings = embeddings[target_type].cpu().numpy()
        np.save(os.path.join(embeddings_dir, f'embeddings_seed_{seed}.npy'), val_embeddings)

        return f1_micro, f1_macro, f1_weigh, auc, prec, rec