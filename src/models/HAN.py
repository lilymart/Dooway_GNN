import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv, Linear


class HAN(torch.nn.Module):

    def __init__(self, metadata, target_type="user", hidden_channels=128, out_channels=2, num_layers=2, heads=4, dropout=0.3):
        super().__init__()

        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads for HANConv")

        self.target_type = target_type
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            HANConv(
                in_channels=-1,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
                dropout=dropout,
            )
        )

        for _ in range(num_layers - 1):
            self.convs.append(
                HANConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=heads,
                    dropout=dropout,
                )
            )

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

            missing = [node_type for node_type, x in x_dict.items() if x is None]
            if missing:
                raise RuntimeError(
                    "HAN could not update node types with no incoming relations: "
                    + ", ".join(missing)
                    + ". Ensure that the heterogeneous graph contains the required reverse relations."
                )

            x_dict = {
                node_type: F.dropout(F.elu(x), p=self.dropout, training=self.training)
                for node_type, x in x_dict.items()
            }

        embeddings = x_dict
        out_dict = {self.target_type: self.classifier(embeddings[self.target_type])}
        return out_dict, embeddings
