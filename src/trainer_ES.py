import os
import torch
import pandas as pd
import copy
import numpy as np
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import torch.nn.functional as F

from src.utils import at_least_k_accuracy


def train_node_classifier(model, data, optimizer, criterion, target_type, multilabel=True, n_epochs=200, patience=50, epsilon=1e-4):

    best_val_f1 = 0.0  # To keep track of the best validation F1 score
    best_model_state = None  # To store the best model's state
    epochs_without_improvement = 0  # To track epochs without improvement
    best_epoch = 0
    epochs_trained = 0

    #loss_values = []

    for epoch in range(1, n_epochs + 1):
        epochs_trained = epoch
        model.train()
        optimizer.zero_grad()
        out, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        mask = data[target_type].train_mask
        loss = criterion(out[target_type][mask], data[target_type].y[mask])
        loss.backward()
        optimizer.step()

        if multilabel:
            report_dict = eval_node_classifier_multilabel(model, data, target_type, split="val")
        else:
            report_dict = eval_node_classifier_multiclass(model, data, target_type, split="val")

        f1_macro = report_dict["macro avg"]["f1-score"]
        f1_weighted = report_dict["weighted avg"]["f1-score"]

        #loss_values.append(loss.item())

        val_score = f1_macro #f1_weighted
        if val_score > best_val_f1 + epsilon:
            best_val_f1 = val_score
            best_model_state = copy.deepcopy(model.state_dict())  # Save the best model state
            best_epoch = epoch
            epochs_without_improvement = 0  # Reset the counter
        else:
            epochs_without_improvement += 1


        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best Val f1_macro: {best_val_f1:.3f}")
            break

        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val f1_macro: {f1_macro:.3f}, Val f1_weighted: {f1_weighted:.3f}')

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    training_info = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_f1,
        "epochs_trained": epochs_trained,
    }

    return model, training_info



def eval_node_classifier_multiclass(model, data, target_type, split="val", report_path=None):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        prob = F.softmax(out[target_type], dim=-1)
        pred = prob.argmax(dim=-1)

        if split == "train":
            mask = data[target_type].train_mask
        elif split == "val":
            mask = data[target_type].val_mask
        elif split == "test":
            mask = data[target_type].test_mask
        else:
            raise ValueError(f"Unknown split {split}")

        y_true = data[target_type].y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        report_dict = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()

        if report_path is not None:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            report_df.to_excel(report_path, index=True)
            print(f"Saved multiclass classification report to {report_path}")

        return report_dict


def eval_node_classifier_multilabel(
    model, data, target_type, tracker_report=None, training_info=None, split="val", report_path=None
):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        prob = torch.sigmoid(out[target_type])
        pred = (prob > 0.5).long()

        if split == "train":
            mask = data[target_type].train_mask
        elif split == "val":
            mask = data[target_type].val_mask
        elif split == "test":
            mask = data[target_type].test_mask
        else:
            raise ValueError(f"Unknown split {split}")

        y_true = data[target_type].y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()

        report_dict = classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        )

        # Extend with multilabel-specific metrics
        report_dict["hamming_loss"] = hamming_loss(y_true, y_pred)
        report_dict["subset_accuracy"] = accuracy_score(y_true, y_pred)

        max_labels_true = int(y_true.sum(axis=1).max())
        for k in range(1, max_labels_true+1):
            key = f"at_least_{k}_labels"
            report_dict[key] = at_least_k_accuracy(y_true, y_pred, k=k)

        # Extends with training info
        if training_info is not None:
            report_dict["epochs_trained"] = training_info["epochs_trained"]
            report_dict["training_time"] = training_info["training_time"]
            report_dict["num_trainable_params"] = training_info["num_trainable_params"]
            report_dict["peak_gpu_memory_mb"] = training_info["peak_gpu_memory_mb"]

        # Extends with FLOPs metrics
        if tracker_report is not None:
            report_dict["FLOPs_bwd"] = tracker_report.model_backward_flop
            report_dict["BOPs_bwd"] = tracker_report.model_backward_bop
            report_dict["FLOPs_fwd"] = tracker_report.model_forward_flop
            report_dict["BOPs_fwd"] = tracker_report.model_forward_bop
            report_dict["FLOPs_tot"] = tracker_report.overall_flop
            report_dict["BOPs_tot"] = tracker_report.overall_bop

        report_df = pd.DataFrame(report_dict).transpose()

        if report_path is not None:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            report_df.to_excel(report_path, index=True)
            print(f"Saved multilabel classification report to {report_path}")

        return report_dict

