import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths
import torch_geometric.transforms as T

from src.utils import get_base_dir, update_labels_from_df

"""
Returns True if both source and destination node types (first and last tokens) are in node_types.
"""
def keep_edge(edge_type, node_types):

    parts = edge_type.split("_")
    if len(parts) < 2:
        return False  # malformed name
    src, dst = parts[0], parts[-1]
    return src in node_types and dst in node_types


def load_heterodata(dataset_name="Basilicata", subset=True, feats_dim=None, use_attrs=True, use_imgs=True, multilabel=True, max_k=8, fixed_test_set=False, experts_test_set=False, seed=42, robustness=100):

    base_dir = os.path.join(get_base_dir(), dataset_name)
    if subset:
        base_dir = os.path.join(base_dir, "subset")
    heterodata_dir = os.path.join(base_dir, "heterodata")
    nodes_dir = os.path.join(heterodata_dir, "node_features", "tensors")
    edges_dir = os.path.join(heterodata_dir, "edgelists", "tensors")
    edge_attrs_dir = os.path.join(heterodata_dir, "edgeattrs", "tensors")

    node_types = ["category", "poi", "review", "service", "user", "image"] #"zone",
    target_type = "user"
    edge_types_files = [f for f in os.listdir(edges_dir) if os.path.isfile(os.path.join(edges_dir, f))]
    all_edge_types = [os.path.splitext(f)[0] for f in edge_types_files]
    edge_types = [e for e in all_edge_types if keep_edge(e, node_types)]

    data = HeteroData()

    # Load node features

    if feats_dim is None:
        suffix = ""
    else:
        suffix = f"_{feats_dim}"

    for ntype in node_types:
        X = torch.load(os.path.join(nodes_dir, f"{ntype}{suffix}.pt"))
        if not use_attrs or (not use_imgs and ntype == "image"):
            X = torch.zeros_like(X)
        data[ntype].x = X
        #print(ntype, "sum =", data[ntype].x.abs().sum().item(), "nonzero =", torch.count_nonzero(data[ntype].x).item())

    if multilabel:
        y = torch.load(os.path.join(heterodata_dir, f"{target_type}s_avg_multilabel_k{max_k}.pt"))  # [num_samples x num_classes]
    else:
        y = torch.load(os.path.join(heterodata_dir, f"{target_type}_labels.pt"))  # [num_samples]

    df_path = os.path.join(heterodata_dir, "output_esperti.parquet")
    y_overwritten, test_ids = update_labels_from_df(y, df_path)
    if experts_test_set:
        data[target_type].y = y_overwritten
    else:
        data[target_type].y = y

    # Load edgelists and optional edge attributes

    for edge_type in edge_types:
        parts = edge_type.split("_")

        src_ntype, etype, tgt_ntype = parts[0], "_".join(parts[1:-1]) if len(parts) > 2 else "", parts[-1]
        edge_index = torch.load(os.path.join(edges_dir, f"{edge_type}.pt"))
        data[src_ntype, etype, tgt_ntype].edge_index = edge_index

        #ptional edge attributes
        edge_attr_path = os.path.join(edge_attrs_dir, f"{edge_type}.pt")
        if os.path.exists(edge_attr_path):

            edge_attr = torch.load(edge_attr_path).float()

            # Scalar attribute:
            # [num_edges] -> [num_edges, 1]
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            data[src_ntype, etype, tgt_ntype].edge_attr = edge_attr


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

    for edge_type in data.edge_types:
        store = data[edge_type]

        if "edge_weight" in store and "edge_attr" not in store:
            store.edge_attr = store.edge_weight.float().view(-1, 1)

    data = _normalize_edge_attributes(data)

    val_ratio = 0.15

    if fixed_test_set:

        num_nodes = data[target_type].num_nodes
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        rng = torch.Generator().manual_seed(seed)

        test_mask[test_ids] = True
        remaining = torch.where(~test_mask)[0]

        # split remaining into train + val
        num_val = int(val_ratio * len(remaining))
        perm = torch.randperm(len(remaining), generator=rng)

        val_nodes = remaining[perm[:num_val]]
        train_nodes = remaining[perm[num_val:]]

        val_mask[val_nodes] = True
        train_mask[train_nodes] = True

        data[target_type].train_mask = train_mask
        data[target_type].val_mask = val_mask
        data[target_type].test_mask = test_mask

    else:

        #transform = T.RandomNodeSplit(num_val=val_ratio, num_test=0.15)
        #data = transform(data)
        #df = pd.read_parquet(os.path.join(base_dir, f"split_flags_{seed}", f"split_flags_{seed}_{robustness}.parquet"))
        df = pd.read_parquet(os.path.join(base_dir, f"split_flags_{seed}.parquet"))
        data[target_type].train_mask = torch.tensor(df["train"].to_numpy(), dtype=torch.bool)
        data[target_type].val_mask = torch.tensor(df["validation"].to_numpy(), dtype=torch.bool)
        data[target_type].test_mask = torch.tensor(df["test"].to_numpy(), dtype=torch.bool)


    return data


"""
Normalize edge attributes relation-wise.
- Standard numeric edge attributes: z = (x - mean) / std
- Metapath/count-based edge attributes: log1p(x) followed by z-score normalization
Normalization is performed independently for each edge type and each edge feature dimension.
"""
def _normalize_edge_attributes(data, eps=1e-8):

    for edge_type in data.edge_types:
        store = data[edge_type]

        if "edge_attr" not in store:
            continue

        edge_attr = store.edge_attr.float()

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)


        # AddMetaPaths(weighted=True) produces edge_weight.
        # These values represent counts / accumulated path weights.
        if "edge_weight" in store:
            edge_attr = torch.log1p(edge_attr)

        # Relation-wise, feature-wise standardization
        mean = edge_attr.mean(dim=0, keepdim=True)
        std = edge_attr.std(dim=0, keepdim=True, unbiased=False)

        # Avoid division by zero for constant attributes
        valid = std > eps
        normalized = torch.zeros_like(edge_attr)
        normalized[:, valid.squeeze(0)] = (edge_attr[:, valid.squeeze(0)] - mean[:, valid.squeeze(0)]) / std[:, valid.squeeze(0)]
        store.edge_attr = normalized

    return data


if __name__ == "__main__":
    region = "Basilicata"
    dim = 256
    data = load_heterodata(dataset_name=region, subset=True, feats_dim=dim, seed=42)
    print(data)