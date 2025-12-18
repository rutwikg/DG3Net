# Author - Rutwik Gulakala - gulakala@iam.rwth-aachen.de

# =============================================================================
# Copyright (c) 2025 Rutwik Gulakala
# All Rights Reserved.
#
# This source code is part of an ongoing research project and is made
# publicly visible for academic transparency and peer review purposes only.
#
# Unauthorized copying, modification, distribution, or use of this code,
# in whole or in part, is strictly prohibited without prior written
# permission from the author.
#
# This code is provided "AS IS", without warranty of any kind.
# =============================================================================

# FEMAT Layer for FEM-Aware Attention

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class FEMAT(MessagePassing):
    """
    FEM-aware attention layer (FEMAT).

    Differences to GATv2Conv:
    - Uses relative positions (pos_j - pos_i)
    - Uses material contrast (mat_j - mat_i)
    - Uses DOF compatibility (dof_i * dof_j)
    - Can be gated by boundary conditions (bc)
    - Optional global features g can be injected
    """
  
    def __init__(
        self,
        in_channels,
        out_channels,
        pos_dim=3,
        dof_dim=6,
        mat_dim=4,
        global_dim=0,
        heads=4,
        concat=True,
        aggr="add"
    ):
        super().__init__(aggr=aggr, node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.global_dim = global_dim

        # Linear projection of node features
        self.lin_h = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Linear projections for geometric & material differences and DOF compatibility
        self.lin_pos = nn.Linear(pos_dim, heads * out_channels, bias=False)
        self.lin_mat = nn.Linear(mat_dim, heads * out_channels, bias=False)
        self.lin_dof = nn.Linear(dof_dim, heads * out_channels, bias=False)

        # Optional projection for global features
        if global_dim > 0:
            self.lin_g = nn.Linear(global_dim, heads * out_channels, bias=False)
        else:
            self.lin_g = None

        # Attention MLP: takes concatenated [h_i, h_j, geo, mat, dof] per head
        att_in_dim = 5 * heads * out_channels  # hi, hj, pos, mat, dof
        if global_dim > 0:
            att_in_dim += heads * out_channels  # + global

        self.att_mlp = nn.Sequential(
            nn.Linear(att_in_dim, att_in_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(att_in_dim // 2, heads)  # one logit per head
        )

        # Optional bias for output
        if concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, edge_index, pos, dof, mat, bc=None, g=None):
        """
        x   : [N, F] node features
        pos : [N, 3] coordinates
        dof : [N, dof_dim] DOF one-hot
        mat : [N, mat_dim] material features
        bc  : [N] or [N,1] binary BC flag (0 = fixed, 1 = free) (optional)
        g   : [global_dim] or [B, global_dim] (optional, single graph assumed)
        """
        H, C = self.heads, self.out_channels

        # Project node features
        h = self.lin_h(x)  # [N, H*C]
        h = h.view(-1, H, C)  # [N, H, C]

        # Pre-store aux features; they are passed to message() via **kwargs
        return self.propagate(
            edge_index,
            x=h, pos=pos, dof=dof, mat=mat, bc=bc, g=g,
            size=None
        )

    def message(self, x_i, x_j, pos_i, pos_j, dof_i, dof_j, mat_i, mat_j, bc_i, bc_j, g, index, ptr, size_i):
        """
        x_i, x_j: [E, H, C]
        pos_i, pos_j: [E, 3]
        dof_i, dof_j: [E, dof_dim]
        mat_i, mat_j: [E, mat_dim]
        bc_i, bc_j: [E] or [E,1]
        g: [global_dim] or None
        """
        H, C = x_i.size(1), x_i.size(2)

        # Relative geometric and material features
        delta_pos = pos_j - pos_i        # [E, 3]
        delta_mat = mat_j - mat_i        # [E, mat_dim]
        dof_comp = dof_i * dof_j         # [E, dof_dim] (DOF compatibility)

        # Project these into same latent space as h
        geo_emb = self.lin_pos(delta_pos).view(-1, H, C)  # [E, H, C]
        mat_emb = self.lin_mat(delta_mat).view(-1, H, C)  # [E, H, C]
        dof_emb = self.lin_dof(dof_comp).view(-1, H, C)   # [E, H, C]

        # Broadcast global features if present
        if self.lin_g is not None and g is not None:
            if g.dim() == 1:
                g = g.unsqueeze(0)      # [1, G]
            g_proj = self.lin_g(g)      # [1, H*C]
            g_proj = g_proj.view(1, H, C).expand(x_i.size(0), H, C)  # [E, H, C]
        else:
            g_proj = torch.zeros_like(x_i)

        # Prepare attention input: concat along last dim
        att_in = torch.cat(
            [x_i, x_j, geo_emb, mat_emb, dof_emb, g_proj],
            dim=-1
        )  # [E, H, 5C (+C if global)]

        # Flatten heads for MLP
        E = att_in.size(0)
        att_in_flat = att_in.view(E, -1)   # [E, H*5C(+...)]
        alpha_logits = self.att_mlp(att_in_flat)  # [E, H]

        # Softmax over neighbours j for each i, per head
        alpha = softmax(alpha_logits, index, num_nodes=size_i)  # [E, H]
        alpha = alpha.unsqueeze(-1)  # [E, H, 1]

        # Directionally weighted message = attention * (x_j + geo_emb)
        msg = alpha * (x_j + geo_emb)  # [E, H, C]

        # Optional: BC gating (e.g., reduce updates for fully constrained nodes)
        if bc_i is not None:
            # Ensure shape [E, 1, 1]
            if bc_i.dim() == 1:
                bc_i = bc_i.unsqueeze(-1)
            bc_gate = bc_i.unsqueeze(-1)  # [E, 1, 1], 0 = fixed, 1 = free
            msg = msg * bc_gate

        return msg

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        # Standard 'add' aggregation over neighbours
        return super().aggregate(inputs, index, ptr, dim_size)

    def update(self, aggr_out):
        """
        aggr_out: [N, H, C]
        """
        H, C = self.heads, self.out_channels

        if self.concat:
            out = aggr_out.view(-1, H * C)  # [N, H*C]
        else:
            out = aggr_out.mean(dim=1)      # [N, C]

        out = out + self.bias
        return out
