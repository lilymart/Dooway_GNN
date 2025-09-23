import torch
from torch_geometric.utils import negative_sampling

from src.unsupervised_learning.contrast import combine_metapaths_fast, build_pos_neg_from_threshold, linkpred_bce_loss



def train_contrastive_gat(
    model, data, metapath_types, metapath_weights=None,
    threshold_percentile=90, epochs=10, lr=1e-3, neg_ratio=1.0, device='cuda'
):
    model = model.to(device)
    data = data.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 1) Build meta-path graph once (fast, vectorized)
    edge_index, edge_weight = combine_metapaths_fast(
        data, metapath_types, metapath_weights, make_undirected=True
    )
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    num_users = data['user'].num_nodes

    # 2) Pos/neg sets
    pos_ei, neg_ei, thr = build_pos_neg_from_threshold(
        edge_index, edge_weight, num_users,
        threshold_percentile=threshold_percentile, neg_ratio=neg_ratio
    )
    print(f"Threshold={thr:.4f} | Pos={pos_ei.size(1)} | Neg={neg_ei.size(1)}")

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()

        z_user, _ = model(data.x_dict, data.edge_index_dict)  # [U, d]

        loss = linkpred_bce_loss(z_user, pos_ei, neg_ei, temperature=0.5)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # Resample negatives periodically (helps contrastive objectives):
        if epoch % 5 == 0:
            neg_ei = negative_sampling(
                edge_index=pos_ei,
                num_nodes=num_users,
                num_neg_samples=neg_ei.size(1),
                method='sparse'
            )

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")

