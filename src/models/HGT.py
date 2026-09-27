import torch
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class HGT(torch.nn.Module):
    """Heterogeneous Graph Transformer baseline for node classification."""

    def __init__(self, metadata, target_type="user", hidden_channels=128, out_channels=2, num_layers=2, heads=4, dropout=0.3):
        super().__init__()

        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads for HGTConv")

        self.node_types = metadata[0]
        self.target_type = target_type
        self.dropout = dropout

        self.input_lins = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in self.node_types
        })

        self.convs = torch.nn.ModuleList([
            HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=heads,
            )
            for _ in range(num_layers)
        ])

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x_dict = {
            node_type: F.relu(self.input_lins[node_type](x))
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            updated = conv(x_dict, edge_index_dict)

            # HGTConv may omit node types that never occur as destinations.
            next_x_dict = {}
            for node_type, x in x_dict.items():
                new_x = updated.get(node_type)
                if new_x is None:
                    new_x = x
                else:
                    new_x = F.elu(new_x)
                next_x_dict[node_type] = F.dropout(
                    new_x, p=self.dropout, training=self.training
                )
            x_dict = next_x_dict

        embeddings = x_dict
        out_dict = {self.target_type: self.classifier(embeddings[self.target_type])}
        return out_dict, embeddings
