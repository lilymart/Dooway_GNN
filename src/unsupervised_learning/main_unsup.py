import os
import torch
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.data_loading import load_heterodata
from src.models.HeteroGAT_unsup import HeteroGAT
from src.unsupervised_learning.trainer_unsup import train_contrastive_gat
from src.utils import get_device, get_base_dir

"""
metapaths = [
    ('user', 'metapath_0', 'user'),
    ('user', 'metapath_1', 'user'),
    ('user', 'metapath_2', 'user'),  # example: maybe more paths
]
"""

data = load_heterodata(dataset_name="Basilicata", subset=True, feats_dim=256)
metapaths = list(data.metapath_dict.keys())
model = HeteroGAT(data.metadata(), hidden_channels=64, out_channels=32, num_layers=3)
device = get_device()
data, model = data.to(device), model.to(device)


train_contrastive_gat(
    model=model,
    data=data,
    metapath_types=metapaths,
    metapath_weights=None,
    threshold_percentile=90,
    epochs=200,
    lr=1e-3,
)

model.eval()
with torch.no_grad():
    z_user, _ = model(data.x_dict, data.edge_index_dict)

# Save embeddings
embeddings = z_user.cpu().numpy()
save_dir = os.path.join(get_base_dir(), "Basilicata", "subset", "embeddings")
#os.makedirs(save_dir, exist_ok=True)
np.save(f"{save_dir}/user_embeddings.npy", embeddings)

# Optional: normalize embeddings for better clustering performance
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Initialize HDBSCAN
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,   # minimum number of points per cluster
    min_samples=5,         # controls sensitivity to noise
    metric='euclidean',    # distance metric
    cluster_selection_method='eom'  # "excess of mass" method
)

# Fit embeddings
cluster_labels = clusterer.fit_predict(embeddings_scaled)

print(f"Found {len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters")
print(f"Number of noise points: {(cluster_labels == -1).sum()}")

# Optional: visualize in 2D using UMAP
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
emb_2d = reducer.fit_transform(embeddings_scaled)

plt.figure(figsize=(8,6))
plt.scatter(emb_2d[:,0], emb_2d[:,1], c=cluster_labels, cmap='tab20', s=5)
plt.colorbar(label='Cluster')
plt.title('HDBSCAN Clustering of User Embeddings')
plt.show()