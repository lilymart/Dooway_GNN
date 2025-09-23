import os
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report #f1_score, roc_auc_score, precision_score, recall_score
import torch.nn.functional as F

from src.utils import parse_classification_report


def train_node_classifier(model, data, optimizer, criterion, target_type, multilabel=True, n_epochs=200, patience=20, epsilon=1e-4):

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

        if multilabel:
            report_dict = eval_node_classifier_multilabel(model, data, target_type, split="val")
        else:
            report_dict = eval_node_classifier_multiclass(model, data, target_type, split="val")

        metrics = parse_classification_report(report_dict)
        f1_macro = metrics["macro avg"]["f1-score"]

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


def eval_node_classifier_multiclass(model, data, target_type, split="val", report_path=None):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x_dict, data.edge_index_dict)
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
    model, data, target_type, split="val", report_path=None
):
    model.eval()
    with torch.no_grad():
        out, _ = model(data.x_dict, data.edge_index_dict)
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
        report_df = pd.DataFrame(report_dict).transpose()

        if report_path is not None:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            report_df.to_excel(report_path, index=True)
            print(f"Saved multilabel classification report to {report_path}")

        return report_dict

