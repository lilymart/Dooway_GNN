import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, Linear

"""One heterogeneous GATv2 layer with relation-specific parameters.

A separate GATv2Conv is instantiated for each edge type. Numerical edge
attributes are used only for relations listed in ``edge_attr_dims``.
Relation-specific outputs targeting the same node type are summed.
"""


class RelationGATv2Layer(torch.nn.Module):

    def __init__(self, metadata, hidden_channels, edge_attr_dims=None, first_layer=False, heads=1, dropout=0.3):
        super().__init__()

        self.node_types, self.edge_types = metadata
        self.edge_types = list(self.edge_types)
        self.edge_attr_dims = edge_attr_dims or {}

        self.convs = torch.nn.ModuleDict()
        self.edge_type_to_key = {}

        in_channels = (-1, -1) if first_layer else (hidden_channels, hidden_channels)

        for i, edge_type in enumerate(self.edge_types):
            key = f"rel_{i}"
            self.edge_type_to_key[edge_type] = key

            self.convs[key] = GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                heads=heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
                edge_dim=self.edge_attr_dims.get(edge_type),
                share_weights=False,
            )

        self.root_lins = torch.nn.ModuleDict()
        root_in_channels = -1 if first_layer else hidden_channels
        for node_type in self.node_types:
            self.root_lins[node_type] = Linear(
                root_in_channels,
                hidden_channels
            )



    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        edge_attr_dict = edge_attr_dict or {}

        messages = {node_type: [] for node_type in self.node_types}

        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue

            src_type, _, dst_type = edge_type
            conv = self.convs[self.edge_type_to_key[edge_type]]
            edge_index = edge_index_dict[edge_type]
            x_pair = (x_dict[src_type], x_dict[dst_type])

            if edge_type in self.edge_attr_dims:
                if edge_type not in edge_attr_dict:
                    raise ValueError(
                        f"Relation {edge_type} was configured with edge attributes "
                        "but no edge_attr tensor was provided."
                    )

                edge_attr = edge_attr_dict[edge_type]
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.view(-1, 1)

                expected_dim = self.edge_attr_dims[edge_type]
                if edge_attr.size(-1) != expected_dim:
                    raise ValueError(
                        f"Relation {edge_type}: expected edge_attr dimension "
                        f"{expected_dim}, got {edge_attr.size(-1)}."
                    )

                edge_attr = edge_attr.to(dtype=x_dict[src_type].dtype)
                out = conv(x_pair, edge_index, edge_attr=edge_attr)
            else:
                # Relations without edge features use standard GATv2 attention.
                out = conv(x_pair, edge_index)

            messages[dst_type].append(out)

        out_dict = {}

        for node_type in self.node_types:
            node_messages = messages[node_type]

            # Sum relation-specific contributions
            rel_out = torch.stack(node_messages, dim=0).sum(dim=0)
            root_out = self.root_lins[node_type](F.relu(x_dict[node_type]))
            out_dict[node_type] = rel_out + root_out

        return out_dict


"""Relation-specific, edge-aware heterogeneous GATv2 encoder.

Edge attributes enter the GATv2 attention computation only for relations
that actually provide them. Relations without edge attributes are processed
by ordinary relation-specific GATv2 attention.
"""
class My_GAT_enhanced(torch.nn.Module):

    uses_edge_attr = True

    def __init__(self, metadata, target_type="user", hidden_channels=64, out_channels=15, dropout=0.3, num_layers=3, heads=1, edge_attr_dims=None):
        super().__init__()

        self.target_type = target_type
        self.dropout = dropout
        self.num_layers = num_layers
        self.edge_attr_dims = edge_attr_dims or {}

        self.convs = torch.nn.ModuleList([
            RelationGATv2Layer(
                metadata=metadata,
                hidden_channels=hidden_channels,
                edge_attr_dims=self.edge_attr_dims,
                first_layer=(layer_idx == 0),
                heads=heads,
                dropout=dropout,
            )
            for layer_idx in range(num_layers)
        ])

        self.classifier = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        h_dict = x_dict

        for layer_idx, conv in enumerate(self.convs):
            h_dict = conv(
                h_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )

            if layer_idx < self.num_layers - 1:
                h_dict = {node_type: F.dropout(F.elu(h), p = self.dropout, training = self.training)
                for node_type, h in h_dict.items()}


        embeddings = h_dict
        out_dict = {
            self.target_type: self.classifier(embeddings[self.target_type])
        }

        return out_dict, embeddings
