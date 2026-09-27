import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.utils import softmax


class SimpleHGNConv(MessagePassing):
    """Simple-HGN attention layer with edge-type and residual attention."""

    def __init__(
        self,
        in_channels,
        out_channels,
        edge_dim,
        heads=4,
        dropout=0.3,
        negative_slope=0.2,
        beta=0.05,
        residual=True,
    ):
        super().__init__(aggr="add", node_dim=0)

        if out_channels % heads != 0:
            raise ValueError("out_channels must be divisible by heads")

        self.heads = heads
        self.head_dim = out_channels // heads
        self.out_channels = out_channels
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.beta = beta
        self.residual = residual

        self.node_lin = Linear(in_channels, out_channels, bias=False)
        self.edge_lin = Linear(edge_dim, out_channels, bias=False)
        self.att = torch.nn.Parameter(torch.empty(1, heads, 3 * self.head_dim))

        if residual:
            self.res_lin = (
                Linear(in_channels, out_channels, bias=False)
                if in_channels != out_channels
                else torch.nn.Identity()
            )
        else:
            self.res_lin = None

        self.bias = torch.nn.Parameter(torch.zeros(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.node_lin.reset_parameters()
        self.edge_lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att)
        if hasattr(self.res_lin, "reset_parameters"):
            self.res_lin.reset_parameters()
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_attr, alpha_prev=None):
        h = self.node_lin(x).view(-1, self.heads, self.head_dim)
        e = self.edge_lin(edge_attr).view(-1, self.heads, self.head_dim)

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        alpha = torch.cat([h_dst, h_src, e], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, dst, num_nodes=x.size(0))

        if alpha_prev is not None:
            alpha = (1.0 - self.beta) * alpha + self.beta * alpha_prev

        alpha_for_message = F.dropout(
            alpha, p=self.dropout, training=self.training
        )

        out = self.propagate(
            edge_index,
            x=h,
            alpha=alpha_for_message,
            size=(x.size(0), x.size(0)),
        )
        out = out.reshape(-1, self.out_channels)

        if self.res_lin is not None:
            out = out + self.res_lin(x)

        out = out + self.bias
        return out, alpha

    def message(self, x_j, alpha):
        return x_j * alpha.unsqueeze(-1)


class SimpleHGN(torch.nn.Module):
    """Simple-HGN baseline adapted to PyG HeteroData input dictionaries."""

    def __init__(
        self,
        metadata,
        target_type="user",
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        heads=4,
        edge_dim=64,
        dropout=0.3,
        negative_slope=0.2,
        beta=0.05,
    ):
        super().__init__()

        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")

        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.target_type = target_type
        self.dropout = dropout

        self.input_lins = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in self.node_types
        })

        self.edge_embedding = torch.nn.Embedding(len(self.edge_types), edge_dim)

        self.convs = torch.nn.ModuleList([
            SimpleHGNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                heads=heads,
                dropout=dropout,
                negative_slope=negative_slope,
                beta=beta,
                residual=True,
            )
            for _ in range(num_layers)
        ])

        self.classifier = Linear(hidden_channels, out_channels)

    def _to_homogeneous(self, x_dict, edge_index_dict):
        projected = {}
        offsets = {}
        node_chunks = []
        offset = 0

        for node_type in self.node_types:
            x = F.relu(self.input_lins[node_type](x_dict[node_type]))
            projected[node_type] = x
            offsets[node_type] = offset
            node_chunks.append(x)
            offset += x.size(0)

        x = torch.cat(node_chunks, dim=0)

        edge_chunks = []
        type_chunks = []

        for edge_type_id, edge_type in enumerate(self.edge_types):
            if edge_type not in edge_index_dict:
                continue

            src_type, _, dst_type = edge_type
            edge_index = edge_index_dict[edge_type]

            global_edge_index = torch.stack([
                edge_index[0] + offsets[src_type],
                edge_index[1] + offsets[dst_type],
            ])
            edge_chunks.append(global_edge_index)
            type_chunks.append(
                torch.full(
                    (edge_index.size(1),),
                    edge_type_id,
                    dtype=torch.long,
                    device=edge_index.device,
                )
            )

        if not edge_chunks:
            raise RuntimeError("SimpleHGN received an empty heterogeneous graph")

        edge_index = torch.cat(edge_chunks, dim=1)
        edge_type_ids = torch.cat(type_chunks, dim=0)
        edge_attr = self.edge_embedding(edge_type_ids)

        return x, edge_index, edge_attr, offsets

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        x, edge_index, edge_attr, offsets = self._to_homogeneous(
            x_dict, edge_index_dict
        )

        alpha = None
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, alpha = conv(x, edge_index, edge_attr, alpha_prev=alpha)
            x = F.elu(x)

        # L2 normalization is part of the Simple-HGN refinement.
        x = F.normalize(x, p=2, dim=-1)

        embeddings = {}
        for node_type in self.node_types:
            start = offsets[node_type]
            end = start + x_dict[node_type].size(0)
            embeddings[node_type] = x[start:end]

        out_dict = {self.target_type: self.classifier(embeddings[self.target_type])}
        return out_dict, embeddings
