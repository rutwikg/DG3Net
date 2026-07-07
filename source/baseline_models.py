"""
baseline_models.py
==================
Verbatim class definitions for the literature baselines used in the DG3Net
comparative study. Extracted from LiteratureCompare.ipynb.

Import this file before torch.load(...) on any of the trained baseline
checkpoints, otherwise unpickling the full-model objects fails with
    AttributeError: Can't get attribute 'MeshGraphNet' on <module '__main__'>

Also usable as a source-of-truth for training baselines from scratch via
train_baselines.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GraphUNet as PyGGraphUNet
from torch_geometric.utils import to_undirected


# ---- shared helper ---------------------------------------------------------

def build_mlp(dims, activation=nn.SELU, activate_last=True):
    """MLP builder used across baselines."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or activate_last:
            layers.append(activation())
    return nn.Sequential(*layers)


# ---- GATv2 baseline (MeshGATv2NodeRegression) ------------------------------

class MeshGATv2NodeRegression(nn.Module):
    """5x GATv2 + FC head + Sigmoid. Matches the notebook definition."""
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4):
        super().__init__()
        self.gat1 = GATv2Conv(input_dim,   hidden_dim // heads, heads=heads, concat=True)
        self.gat2 = GATv2Conv(hidden_dim,  hidden_dim // heads, heads=heads, concat=True)
        self.gat3 = GATv2Conv(hidden_dim,  hidden_dim // heads, heads=heads, concat=True)
        self.gat4 = GATv2Conv(hidden_dim,  hidden_dim // heads, heads=heads, concat=True)
        self.gat5 = GATv2Conv(hidden_dim,  hidden_dim // heads, heads=heads, concat=True)

        self.fc = nn.Linear(16, output_dim)
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, hidden_dim)
        self.lin6 = nn.Linear(hidden_dim, 64)
        self.lin7 = nn.Linear(64, 32)
        self.lin8 = nn.Linear(32, 16)
        self.act = nn.SELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = self.gat1(x, ei)
        x = self.gat2(x, ei)
        x = self.gat3(x, ei)
        x = self.gat4(x, ei); x = self.act(self.lin4(x))
        x = self.gat5(x, ei); x = self.act(self.lin5(x))
        x = self.act(self.lin6(x))
        x = self.act(self.lin7(x))
        x = self.act(self.lin8(x))
        return self.sigmoid(self.fc(x))


# ---- MeshGraphNet ----------------------------------------------------------

class GraphNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])
        self.node_mlp = build_mlp([2 * hidden_dim, hidden_dim, hidden_dim])

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        e_input  = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        new_edge = self.edge_mlp(e_input) + edge_attr
        agg = torch.zeros(x.size(0), new_edge.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(new_edge), new_edge)
        n_input = torch.cat([x, agg], dim=-1)
        new_x = self.node_mlp(n_input) + x
        return new_x, new_edge


class MeshGraphNet(nn.Module):
    def __init__(self, input_dim=4, edge_dim=8, hidden_dim=64,
                 output_dim=3, num_mp_steps=15):
        super().__init__()
        self.node_encoder = build_mlp([input_dim, hidden_dim, hidden_dim])
        self.edge_encoder = build_mlp([edge_dim,  hidden_dim, hidden_dim])
        self.processor = nn.ModuleList(
            [GraphNetBlock(hidden_dim) for _ in range(num_mp_steps)]
        )
        self.decoder = nn.Sequential(
            build_mlp([hidden_dim, hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for block in self.processor:
            x, e = block(x, data.edge_index, e)
        return self.decoder(x)


# ---- X-MeshGraphNet --------------------------------------------------------

try:
    from torch_scatter import scatter_mean
except ImportError:
    from torch_geometric.utils import scatter
    def scatter_mean(src, index, dim=0, dim_size=None):
        return scatter(src, index, dim=dim, reduce='mean', dim_size=dim_size)


class XGraphNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])
        self.node_mlp = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])   # + global
        self.global_mlp = build_mlp([2 * hidden_dim, hidden_dim, hidden_dim])

    def forward(self, x, edge_index, edge_attr, u, batch):
        src, dst = edge_index
        e_input  = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        new_edge = self.edge_mlp(e_input) + edge_attr
        agg = torch.zeros(x.size(0), new_edge.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(new_edge), new_edge)

        u_expand = u[batch]                                # broadcast global to nodes
        n_input  = torch.cat([x, agg, u_expand], dim=-1)
        new_x    = self.node_mlp(n_input) + x

        # global update from aggregated node state
        node_mean = scatter_mean(new_x, batch, dim=0, dim_size=u.size(0))
        g_input   = torch.cat([u, node_mean], dim=-1)
        new_u     = self.global_mlp(g_input) + u
        return new_x, new_edge, new_u


class XMeshGraphNet(nn.Module):
    def __init__(self, input_dim=4, edge_dim=8, hidden_dim=64,
                 output_dim=3, num_mp_steps=15):
        super().__init__()
        self.node_encoder   = build_mlp([input_dim, hidden_dim, hidden_dim])
        self.edge_encoder   = build_mlp([edge_dim,  hidden_dim, hidden_dim])
        self.global_init    = nn.Parameter(torch.zeros(hidden_dim))
        self.processor = nn.ModuleList(
            [XGraphNetBlock(hidden_dim) for _ in range(num_mp_steps)]
        )
        self.decoder = nn.Sequential(
            build_mlp([hidden_dim, hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        n_graphs = int(batch.max().item()) + 1
        u = self.global_init.unsqueeze(0).expand(n_graphs, -1).contiguous()
        for block in self.processor:
            x, e, u = block(x, data.edge_index, e, u, batch)
        return self.decoder(x)


# ---- EA-GNN ----------------------------------------------------------------

class EAGraphNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.local_edge_mlp = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])
        self.aug_edge_mlp   = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])
        self.node_mlp       = build_mlp([3 * hidden_dim, hidden_dim, hidden_dim])

    def _process(self, x, edge_index, edge_attr, mlp):
        src, dst = edge_index
        e_in     = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        new_edge = mlp(e_in) + edge_attr
        agg = torch.zeros(x.size(0), new_edge.size(1), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(new_edge), new_edge)
        return new_edge, agg

    def forward(self, x, edge_index, edge_attr, edge_index_aug, edge_attr_aug):
        ne_local, agg_local = self._process(x, edge_index,     edge_attr,     self.local_edge_mlp)
        ne_aug,   agg_aug   = self._process(x, edge_index_aug, edge_attr_aug, self.aug_edge_mlp)
        n_input = torch.cat([x, agg_local + agg_aug, agg_local], dim=-1)
        new_x   = self.node_mlp(n_input) + x
        return new_x, ne_local, ne_aug


class EAGNN(nn.Module):
    def __init__(self, input_dim=4, edge_dim=8, aug_edge_dim=4,
                 hidden_dim=64, output_dim=3, num_mp_steps=10):
        super().__init__()
        self.node_encoder       = build_mlp([input_dim,    hidden_dim, hidden_dim])
        self.local_edge_encoder = build_mlp([edge_dim,     hidden_dim, hidden_dim])
        self.aug_edge_encoder   = build_mlp([aug_edge_dim, hidden_dim, hidden_dim])
        self.processor = nn.ModuleList(
            [EAGraphNetBlock(hidden_dim) for _ in range(num_mp_steps)]
        )
        self.decoder = nn.Sequential(
            build_mlp([hidden_dim, hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        e = self.local_edge_encoder(data.edge_attr)
        ea = self.aug_edge_encoder(data.edge_attr_aug)
        for block in self.processor:
            x, e, ea = block(x, data.edge_index, e, data.edge_index_aug, ea)
        return self.decoder(x)


# ---- GraphUNet -------------------------------------------------------------

class GraphUNet(nn.Module):
    """Wrapper around PyG's GraphUNet with input/output MLPs."""
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3,
                 depth=3, pool_ratio=0.5):
        super().__init__()
        self.input_proj = build_mlp([input_dim, hidden_dim, hidden_dim])
        self.unet = PyGGraphUNet(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            depth=depth,
            pool_ratios=pool_ratio,
            act=F.selu,
        )
        self.output_proj = nn.Sequential(
            build_mlp([hidden_dim, hidden_dim]),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x = self.input_proj(data.x)
        x = self.unet(x, data.edge_index, data.batch)
        return self.output_proj(x)


# ---- edge_attr utilities (needed at inference/training time) ---------------

def compute_edge_attr_full(data_list):
    """8-dim edge attr: [length, rel_x, rel_y, rel_z, rel_thickness, dir_x, dir_y, dir_z]"""
    out = []
    for d in data_list:
        d2 = d.clone()
        src, dst = d2.edge_index
        pos       = d2.x[:, :3]
        rel       = pos[dst] - pos[src]
        length    = rel.norm(dim=1, keepdim=True).clamp_min(1e-12)
        norm_dir  = rel / length
        thick     = d2.x[:, 3:4]
        rel_thick = thick[dst] - thick[src]
        d2.edge_attr = torch.cat([length, rel, rel_thick, norm_dir], dim=1)
        out.append(d2)
    return out


def add_augmented_edges(data_list, k_aug=16):
    """kNN augmented edges for EA-GNN, with [dist, dx, dy, dz] attributes."""
    def knn_pytorch(pos, k):
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        d2   = (diff ** 2).sum(dim=-1); d2.fill_diagonal_(float('inf'))
        _, idx = d2.topk(k, dim=-1, largest=False)
        N = pos.size(0)
        src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k).reshape(-1)
        dst = idx.reshape(-1)
        return torch.stack([src, dst], dim=0)

    out = []
    for d in data_list:
        d2  = d.clone()
        pos = d2.x[:, :3].float()
        ei_aug = knn_pytorch(pos, k=k_aug)
        ei_aug = to_undirected(ei_aug)
        src_a, dst_a = ei_aug
        rel_a  = pos[dst_a] - pos[src_a]
        dist_a = rel_a.norm(dim=1, keepdim=True)
        dir_a  = rel_a / (dist_a + 1e-12)
        d2.edge_index_aug = ei_aug
        d2.edge_attr_aug  = torch.cat([dist_a, dir_a], dim=1)
        out.append(d2)
    return out
