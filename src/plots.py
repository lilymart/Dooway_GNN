import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_labels_distribution(data, output_dir):

    y = data["user"].y

    # CASE 1: single-label classification
    if y.dim() == 1:
        labels = y.numpy()
        num_classes = labels.max().item() + 1
        counts = [(labels == i).sum() for i in range(num_classes)]

    # CASE 2: multi-label classification
    else:
        num_classes = y.size(1)
        counts = y.sum(dim=0).numpy()

    # Create the plot
    plt.figure(figsize=(12, 5))
    plt.bar(range(num_classes), counts, color="royalblue")

    plt.xlabel("Class ID")
    plt.ylabel("Count")
    #plt.title("User Class Distribution")

    plt.xticks(range(num_classes))  # <- show 0,1,2,...14

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,"label_distribution.png"), dpi=300, bbox_inches='tight')
    plt.show()


"""
df = the SAME dataframe used in process_master_file (must contain 'robustness' and metric columns)
"""
def plot_robustness_curves(df, output_dir):

    # --------------------------
    # 1. Group by robustness
    # --------------------------
    grouped = df.groupby("robustness")

    robustness_vals = sorted(df["robustness"].unique())

    # --------------------------
    # 2. Metrics to plot
    # --------------------------
    metrics = [
        "f1_macro_mean",
        "f1_weighted_mean"
    ]

    for metric in metrics:

        means = []
        stds = []

        for r in robustness_vals:
            group = grouped.get_group(r)

            values = group[metric].dropna().astype(float).values

            means.append(group[metric].values[0])

            std_col = metric.replace("_mean", "_std")
            stds.append(group[std_col].values[0])

        # --------------------------
        # 3. Plot
        # --------------------------
        plt.figure()

        plt.errorbar(
            robustness_vals,
            means,
            yerr=stds,
            marker='o',
            capsize=5
        )

        plt.xlabel("Robustness")
        metric_str = metric.replace("_mean", "")
        plt.ylabel(metric_str)
        #plt.title(f"{metric} vs robustness")

        plt.grid(True)

        # Save
        plt.savefig(os.path.join(output_dir, f"{metric_str}_robustness_curve.png"))
        plt.close()


if __name__ == '__main__':

    dir = "/home/martirano/data/dooway/Basilicata/subset/results"
    df = pd.read_excel(os.path.join(dir, "table_metrics_robustness.xlsx"))
    plot_robustness_curves(df, dir)