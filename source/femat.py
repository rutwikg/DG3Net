"""
femat.py
========
The FEMAT (FEM-aware Attention) operator, implemented as a PyG MessagePassing
layer matching Eq. (femat-attn), (femat-message), (femat-update) in the paper.

Attention conditions on THREE explicit FE-derived edge signals in addition to
the endpoint latents:

    * relative geometric direction   Δx_ij  = x_j - x_i       (3-vec)
    * material contrast              Δm_ij  = m_j - m_i       (m_dim-vec, e.g. thickness)
    * DOF compatibility              d_ij   = d_i ⊙ d_j       (dof_dim-vec)

A boundary-condition binary gate BC_i modulates the residual update:
    h_i^{k+1} = BN( BC_i * u_i^{k} + (1 - BC_i) * h_i^{k} )

The layer expects the following on the input `data`:
    data.x                : [N, node_feat_dim]  -- node latents fed to attention
    data.edge_index       : [2, E]
    data.pos              : [N, 3]              -- for Δx computation (physical or nondim)
    data.mat              : [N, m_dim]          -- material/section attributes  (optional)
    data.dof              : [N, dof_dim]        -- one-hot active DOF vector    (optional)
    data.bc               : [N, 1]              -- 1 = free, 0 = constrained    (optional)

If any of `mat`, `dof`, `bc` are absent the corresponding term is zeroed out
and the layer degrades gracefully. Set the ablation-time knobs accordingly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as scatter_softmax


class FEMATConv(MessagePassing):
    """
    Multi-head FEMAT attention.
    In / out dims are per-node latent widths. Total width = hidden_dim.
    Per-head width = hidden_dim // heads.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        pos_dim: int = 3,
        mat_dim: int = 1,     # thickness in the current data.x setup
        dof_dim: int = 6,     # 3 trans + 3 rot
        negative_slope: float = 0.2,
        use_direction_msg: bool = True,
    ):
        super().__init__(aggr="add", node_dim=0)
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.heads = heads
        self.head_dim = out_channels // heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pos_dim = pos_dim
        self.mat_dim = mat_dim
        self.dof_dim = dof_dim
        self.negative_slope = negative_slope
        self.use_direction_msg = use_direction_msg

        # Node latent projections (Q and K share weights per GATv2 convention)
        self.W_h = nn.Linear(in_channels, out_channels, bias=False)
        # Value projection
        self.W_v = nn.Linear(in_channels, out_channels, bias=False)

        # FE-signal projections into attention pre-activation space
        self.W_x = nn.Linear(pos_dim, out_channels, bias=False)          # Δx
        self.W_m = nn.Linear(mat_dim, out_channels, bias=False)          # Δm
        self.W_d = nn.Linear(dof_dim, out_channels, bias=False)          # d_ij

        # Directional message projection (adds Δx into the message)
        if use_direction_msg:
            self.W_g = nn.Linear(pos_dim, out_channels, bias=False)

        # Per-head attention vector a
        self.att = nn.Parameter(torch.empty(heads, self.head_dim))

        # Output linear
        self.W_o = nn.Linear(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

        self._reset()

    def _reset(self):
        for m in [self.W_h, self.W_v, self.W_x, self.W_m, self.W_d, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        if hasattr(self, "W_g"):
            nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, pos, mat=None, dof=None, bc=None):
        """
        x        : [N, in_channels]
        pos      : [N, pos_dim]
        mat      : [N, mat_dim]   or None
        dof      : [N, dof_dim]   or None
        bc       : [N, 1]         or None
        """
        N = x.size(0)
        H, D = self.heads, self.head_dim

        h = self.W_h(x).view(N, H, D)             # [N, H, D]
        v = self.W_v(x).view(N, H, D)             # [N, H, D]

        # zero-pad missing signals
        if mat is None:
            mat = torch.zeros(N, self.mat_dim, device=x.device, dtype=x.dtype)
        if dof is None:
            dof = torch.zeros(N, self.dof_dim, device=x.device, dtype=x.dtype)

        out = self.propagate(
            edge_index,
            h=h, v=v, pos=pos, mat=mat, dof=dof,
        )                                          # [N, H, D]
        out = out.reshape(N, self.out_channels)   # [N, out_channels]
        out = self.W_o(out)

        # Boundary-condition gate + residual (BC=1 free -> take new; BC=0 fixed -> keep prev)
        if bc is None:
            gated = out
        else:
            bc = bc.view(N, 1).to(out.dtype)
            # residual only makes sense if in_channels == out_channels; otherwise fall back
            if x.shape[1] == out.shape[1]:
                gated = bc * out + (1.0 - bc) * x
            else:
                gated = out  # first block; no residual dimension match
        return self.bn(gated)

    def message(self, h_i, h_j, v_j, pos_i, pos_j, mat_i, mat_j, dof_i, dof_j, index):
        """
        h_i, h_j, v_j : [E, H, D]
        pos_i, pos_j  : [E, pos_dim]
        mat_i, mat_j  : [E, mat_dim]
        dof_i, dof_j  : [E, dof_dim]
        index         : destination-node index per edge (for softmax over neighbours of i)
        """
        E, H, D = h_i.shape

        dx = pos_j - pos_i                                        # [E, pos_dim]
        dm = mat_j - mat_i                                        # [E, mat_dim]
        d_ij = dof_i * dof_j                                      # [E, dof_dim]  (Hadamard)

        # Build pre-activation for attention scoring, GATv2-style
        pre = h_i + h_j                                           # [E, H, D]
        pre = pre + self.W_x(dx).view(E, H, D)
        pre = pre + self.W_m(dm).view(E, H, D)
        pre = pre + self.W_d(d_ij).view(E, H, D)
        pre = F.leaky_relu(pre, self.negative_slope)              # [E, H, D]

        # α_ij = softmax_j∈N(i) (a^T pre)
        alpha = (pre * self.att.view(1, H, D)).sum(dim=-1)        # [E, H]
        alpha = scatter_softmax(alpha, index=index)               # softmax over neighbours of i

        msg = v_j                                                 # [E, H, D]
        if self.use_direction_msg:
            msg = msg + self.W_g(dx).view(E, H, D)
        return alpha.unsqueeze(-1) * msg                          # [E, H, D]
