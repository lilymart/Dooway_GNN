import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce, remove_self_loops, negative_sampling


# build homogeneous meta-path-based user graph
@torch.no_grad()
def combine_metapaths_fast(data, metapath_types, metapath_weights=None, make_undirected=True):
    """Return a coalesced user-user edge_index, edge_weight summed across meta-paths."""
    num_users = data['user'].num_nodes
    eidx_list, w_list = [], []

    for i, et in enumerate(metapath_types):
        ei = data[et].edge_index           # [2, E_i]
        w  = data[et].edge_weight.float()  # [E_i]
        alpha = (metapath_weights[i] if metapath_weights is not None else 1.0)
        eidx_list.append(ei)
        w_list.append(w * alpha)

    edge_index = torch.cat(eidx_list, dim=1)      # [2, E_total]
    edge_weight = torch.cat(w_list, dim=0)        # [E_total]

    if make_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

    # Merge duplicates by summing weights:
    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_users, reduce='sum')
    # Drop self-edges:
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    return edge_index, edge_weight


@torch.no_grad()
def build_pos_neg_from_threshold(edge_index, edge_weight, num_users,
                                 threshold_percentile=90, neg_ratio=1):
    # GPU-safe percentile:
    thr = torch.quantile(edge_weight, threshold_percentile / 100.0)

    mask = edge_weight >= thr
    pos_edge_index = edge_index[:, mask]  # [2, P]
    pos_edge_index, _ = remove_self_loops(pos_edge_index)

    # Use PyG negative sampling to get in-graph hard negatives:
    num_pos = pos_edge_index.size(1)
    num_neg = max(1, int(num_pos * neg_ratio))
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_users,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return pos_edge_index, neg_edge_index, thr.item()


def linkpred_bce_loss(z, pos_edge_index, neg_edge_index, temperature=0.5):
    # Normalize for cosine-like scoring:
    z = F.normalize(z, p=2, dim=-1)

    def score(ei):
        # dot product; cosine since z is normalized
        return (z[ei[0]] * z[ei[1]]).sum(dim=-1) / temperature

    pos = score(pos_edge_index)   # [P]
    neg = score(neg_edge_index)   # [N]

    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)

    return F.binary_cross_entropy_with_logits(logits, labels)
