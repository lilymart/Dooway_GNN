import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T

from src.utils import get_base_dir


def load_heterodata(dataset_name="Basilicata", subset=True, feats_dim=None, multilabel=True, seed=42): #TODO: seed

    base_dir = os.path.join(get_base_dir(), dataset_name)
    if subset:
        base_dir = os.path.join(base_dir, "subset")
    base_dir = os.path.join(base_dir, "heterodata")
    nodes_dir = os.path.join(base_dir, "features", "tensors")
    edges_dir = os.path.join(base_dir, "edgelists", "tensors")

    node_types = ["category", "image", "poi", "review", "service", "user"]
    target_type = "user"
    edge_types_files = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    edge_types = [os.path.splitext(f)[0] for f in edge_types_files]

    data = HeteroData()

    # Load node features

    if feats_dim is None:
        suffix = ""
    else:
        suffix = f"_{feats_dim}"

    for ntype in node_types:
        X = torch.load(os.path.join(nodes_dir, f"{ntype}{suffix}.pt"))
        data[ntype].x = X

    if multilabel:
        data[target_type].y = torch.load(os.path.join(base_dir, f"{target_type}_labels_multi.pt"))  # ]num_samples x num_classes]
    else:
        data[target_type].y = torch.load(os.path.join(base_dir, f"{target_type}_labels.pt"))  # [num_samples]

    # Load edgelists

    for edge_type in edge_types:
        parts = edge_type.split("_")

        src_ntype, etype, tgt_ntype = parts[0], "_".join(parts[1:-1]) if len(parts) > 2 else "", parts[-1]
        edgelist = torch.load(os.path.join(edges_dir, f"{edge_type}.pt"))
        data[src_ntype, etype, tgt_ntype].edge_index = edgelist

    # meta-paths
    metapaths = [[('user', 'writes', 'review'),
                  ('review', 'is_about', 'poi'),
                  ('poi', 'is_reviewed_by', 'review'),
                  ('review', 'is_written_by', 'user')],  # URPRU
                 [('user', 'writes', 'review'),
                  ('review', 'is_about', 'service'),
                  ('service', 'is_covered_by', 'review'),
                  ('review', 'is_written_by', 'user')]]  # URSRU
    data = AddMetaPaths(metapaths, weighted=True)(data)

    # train-val-test set
    transform = T.RandomNodeSplit(num_val=0.10, num_test=0.15)
    data = transform(data)

    return data


if __name__ == "__main__":
    region = "Basilicata"
    dim = 256
    data = load_heterodata(dataset_name=region, subset=True, feats_dim=dim)
    print(data)