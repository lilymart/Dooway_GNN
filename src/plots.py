import os
import matplotlib.pyplot as plt
import numpy as np



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