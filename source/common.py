"""
common.py
=========
Shared components used by every ablation script:
    - MeshGATv2NodeRegression (V0 baseline architecture, copied from user's code)
    - MeshFEMATNodeRegression (V4 architecture: FEMAT in first block + GATv2 in rest)
    - Geometry-aware normalization (per-graph, distinct from min-max)
    - Displacement-gradient consistency loss (Sobolev-style, Frobenius on Green-Lagrange)
    - Constitutive consistency regularizer  (only meaningful when output_dim >= 21)
    - Data loading + split utilities

Design principle: variants V0..V4 all use the same forward code path,
switching physics losses and the first block via boolean flags.
"""
import os, glob, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from femat import FEMATConv


# =============================================================================
# EDGE AUGMENTATION (borrowed from EA-GNN style; used by DG3Net V4/V4a variants)
# =============================================================================

def _knn_pytorch(pos, k):
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    d2   = (diff ** 2).sum(dim=-1); d2.fill_diagonal_(float("inf"))
    _, idx = d2.topk(k, dim=-1, largest=False)
    N = pos.size(0)
    src = torch.arange(N, device=pos.device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = idx.reshape(-1)
    return torch.stack([src, dst], dim=0)


def add_augmented_edges_to_dataset(data_list, k_aug=16):
    """
    Attach kNN augmented edges to each graph in-place semantics via clone.
    Adds:
        d.edge_index_aug : [2, E_aug]
        d.edge_attr_aug  : [E_aug, 4]  (dist, dir_x, dir_y, dir_z)
    Uses data.x[:, :3] as position source (same convention as EA-GNN pipeline).
    """
    from torch_geometric.utils import to_undirected
    out = []
    for d in data_list:
        d2  = d.clone()
        pos = d2.x[:, :3].float()
        ei_aug = _knn_pytorch(pos, k=k_aug)
        ei_aug = to_undirected(ei_aug)
        src_a, dst_a = ei_aug
        rel_a  = pos[dst_a] - pos[src_a]
        dist_a = rel_a.norm(dim=1, keepdim=True)
        dir_a  = rel_a / (dist_a + 1e-12)
        d2.edge_index_aug = ei_aug
        d2.edge_attr_aug  = torch.cat([dist_a, dir_a], dim=1)
        out.append(d2)
    return out


# =============================================================================
# STRAIN COMPUTATION (as-is from user's notebook, edge-wise directional gradient)
# =============================================================================

def compute_strain(disp, pos, edge_index, eps=1e-6):
    """
    Edge-directional displacement gradient magnitude.
    disp : [N, 3]   predicted or true displacement
    pos  : [N, 3]   coordinates
    Returns: [E] projected gradients along each edge direction.
    """
    if edge_index.numel() == 0:
        return disp.new_zeros(0)
    src, dst = edge_index
    pos_diff = pos[dst] - pos[src]
    length = pos_diff.norm(dim=1, keepdim=True).clamp_min(eps)
    dir_vec = pos_diff / length
    disp_diff = disp[dst] - disp[src]
    return (disp_diff * dir_vec).sum(dim=1) / length.squeeze(1)


def compute_lambda_adaptive(pred_disp, true_disp, pos, edge_index, criterion, offset=1.0, eps=1e-12):
    """
    User's adaptive-lambda scheme. Returns a scalar tensor lambda_E for the current batch.

    Scales lambda so that lambda * L_strain lands `offset` orders of magnitude
    below L_disp (i.e. it's a soft magnitude match).

    pred_disp, true_disp : [N, 3]
    pos                  : [N, 3]
    edge_index           : [2, E]
    criterion            : e.g. nn.MSELoss()
    """
    l_disp = criterion(pred_disp, true_disp)

    pred_strain = compute_strain(pred_disp, pos, edge_index, eps=1e-6)
    true_strain = compute_strain(true_disp, pos, edge_index, eps=1e-6)

    if pred_strain.numel() == 0 or true_strain.numel() == 0:
        return pred_disp.new_zeros(())

    l_strain = criterion(pred_strain, true_strain)

    exp_disp   = torch.floor(torch.log10(l_disp   + eps))
    exp_strain = torch.floor(torch.log10(l_strain + eps))

    lam = torch.pow(torch.tensor(10.0, device=pred_disp.device),
                    exp_disp - exp_strain + offset)
    lam = torch.clamp(lam, 1e-16, 1e10)
    return lam.detach()  # not part of the graph — used only as a scalar multiplier


# =============================================================================
# BASELINE ARCHITECTURE  (matches the user's MeshGATv2NodeRegression)
# =============================================================================

class GATv2Backbone(nn.Module):
    """
    5 GATv2 layers + decoder head. Identical wiring to user's notebook.
    First block is either GATv2 (V0..V3) or FEMAT (V4).

    When `use_femat_first=True` AND `use_aug_edges=True` AND the input data
    carries `edge_index_aug`, FEMAT runs twice — once on mesh edges, once on
    augmented edges — and outputs are summed. This applies the FEMAT attention
    formulation on the same parallel-path augmented-edge scheme as EA-GNN.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, heads=4,
                 use_femat_first=False,
                 use_aug_edges=False,
                 pos_dim=3, mat_dim=1, dof_dim=6):
        super().__init__()
        self.use_femat_first = use_femat_first
        self.use_aug_edges = use_aug_edges
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if use_femat_first:
            self.gat1 = FEMATConv(
                in_channels=input_dim, out_channels=hidden_dim, heads=heads,
                pos_dim=pos_dim, mat_dim=mat_dim, dof_dim=dof_dim,
            )
            if use_aug_edges:
                # Second FEMAT operating on augmented edges; outputs summed with mesh FEMAT.
                self.gat1_aug = FEMATConv(
                    in_channels=input_dim, out_channels=hidden_dim, heads=heads,
                    pos_dim=pos_dim, mat_dim=mat_dim, dof_dim=dof_dim,
                )
            else:
                self.gat1_aug = None
        else:
            self.gat1 = GATv2Conv(input_dim, hidden_dim // heads, heads=heads, concat=True)
            self.gat1_aug = None

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat3 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat4 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat5 = GATv2Conv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)

        # Decoder head (mirror of user's) — but width-1 final projection to output_dim
        self.lin4 = nn.Linear(hidden_dim, hidden_dim)
        self.lin5 = nn.Linear(hidden_dim, hidden_dim)
        self.lin6 = nn.Linear(hidden_dim, 64)
        self.lin7 = nn.Linear(64, 32)
        self.lin8 = nn.Linear(32, 16)
        self.fc = nn.Linear(16, output_dim)

        self.act = nn.SELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x = data.x
        ei = data.edge_index

        if self.use_femat_first:
            pos = data.pos if hasattr(data, "pos") and data.pos is not None else data.x[:, :3]
            mat = data.mat if hasattr(data, "mat") and data.mat is not None else data.x[:, 3:4]
            dof = data.dof if hasattr(data, "dof") and data.dof is not None else None
            bc  = data.bc  if hasattr(data, "bc")  and data.bc  is not None else None

            x_mesh = self.gat1(x, ei, pos=pos, mat=mat, dof=dof, bc=bc)

            if (self.gat1_aug is not None and hasattr(data, "edge_index_aug")
                    and data.edge_index_aug is not None):
                x_aug = self.gat1_aug(x, data.edge_index_aug, pos=pos, mat=mat, dof=dof, bc=bc)
                x = x_mesh + x_aug
            else:
                x = x_mesh
        else:
            x = self.gat1(x, ei)

        x = self.gat2(x, ei)
        x = self.gat3(x, ei)
        x = self.gat4(x, ei); x = self.act(self.lin4(x))
        x = self.gat5(x, ei); x = self.act(self.lin5(x))
        x = self.act(self.lin6(x))
        x = self.act(self.lin7(x))
        x = self.act(self.lin8(x))
        x = self.fc(x)
        return self.sigmoid(x)


# =============================================================================
# NORMALIZATION MODES
# =============================================================================

def apply_minmax(data_list, stats):
    """Standard dataset-wide min-max normalization (V0)."""
    out = []
    x_min = stats["input_stats"]["min"]
    x_max = stats["input_stats"]["max"]
    y_min = stats["output_stats"]["min"]
    y_max = stats["output_stats"]["max"]
    for d in data_list:
        d = d.clone()
        d.x = (d.x - x_min[:d.x.shape[1]]) / (x_max[:d.x.shape[1]] - x_min[:d.x.shape[1]] + 1e-12)
        y = d.y
        d.y = (y - y_min[:y.shape[-1]]) / (y_max[:y.shape[-1]] - y_min[:y.shape[-1]] + 1e-12)
        out.append(d)
    return out


def apply_geom_aware(data_list, stats, coord_idx=(0, 1, 2)):
    """
    Geometry-aware normalization:
        - coordinates: per-graph, subtract centroid, divide by diameter L_ref
        - remaining node features: dataset min-max
        - targets:                dataset min-max
    Stores L_ref on data.L_ref for downstream loss scaling if needed.
    """
    out = []
    cidx = list(coord_idx)
    other_idx = None
    x_min = stats["input_stats"]["min"]
    x_max = stats["input_stats"]["max"]
    y_min = stats["output_stats"]["min"]
    y_max = stats["output_stats"]["max"]
    for d in data_list:
        d = d.clone()
        pos = d.x[:, cidx].float()
        centroid = pos.mean(dim=0, keepdim=True)
        pos_c = pos - centroid
        L_ref = float(pos_c.norm(dim=1).max().clamp_min(1e-12))
        pos_n = pos_c / L_ref

        x_new = d.x.clone()
        x_new[:, cidx] = pos_n

        # min-max the rest
        if other_idx is None:
            other_idx = [i for i in range(d.x.shape[1]) if i not in cidx]
        if len(other_idx) > 0:
            xmin_o = x_min[other_idx]
            xmax_o = x_max[other_idx]
            x_new[:, other_idx] = (x_new[:, other_idx] - xmin_o) / (xmax_o - xmin_o + 1e-12)
        d.x = x_new

        y = d.y
        d.y = (y - y_min[:y.shape[-1]]) / (y_max[:y.shape[-1]] - y_min[:y.shape[-1]] + 1e-12)

        d.L_ref = torch.tensor(L_ref)
        d.pos = pos_n  # exposed for FEMAT

        out.append(d)
    return out


# =============================================================================
# LOSSES
# =============================================================================

def displacement_gradient_loss(pred_disp, true_disp, pos, edge_index):
    """
    Sobolev-style loss on the *edge-wise* displacement gradient.

    Computes an edge-relative displacement gradient magnitude for both prediction
    and ground truth, then MSE between them. Matches the paper's
    "displacement-gradient consistency loss" formulation and the compute_strain
    utility in the user's original notebook, but with axis-safe divisor.

    pred_disp, true_disp : [N, 3]
    pos                  : [N, 3]  (nondim or physical)
    edge_index           : [2, E]
    """
    if edge_index.numel() == 0:
        return pred_disp.new_zeros(())
    src, dst = edge_index
    pos_diff = pos[dst] - pos[src]                         # [E, 3]
    length = pos_diff.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dir_vec = pos_diff / length

    p_diff = pred_disp[dst] - pred_disp[src]               # [E, 3]
    t_diff = true_disp[dst] - true_disp[src]

    # projected gradient along the edge direction
    p_grad = (p_diff * dir_vec).sum(dim=1) / length.squeeze(1)
    t_grad = (t_diff * dir_vec).sum(dim=1) / length.squeeze(1)
    return F.mse_loss(p_grad, t_grad)


def constitutive_loss(pred, true, E_modulus=70e3, nu=0.33):
    """
    Constitutive-consistency regularizer applied only when the output tensor
    contains stress, strain, and plastic-strain channels.

    Convention (matches user's y_features order):
        pred/true[:, 0:3]     : u, v, w                       (displacements)
        pred/true[:, 3:9]     : S_xx, S_yy, S_zz, S_xy, S_yz, S_xz    (stress, Voigt)
        pred/true[:, 9:15]    : LE_xx, LE_yy, LE_zz, LE_xy, LE_yz, LE_xz  (total strain)
        pred/true[:, 15:21]   : PE_xx, PE_yy, PE_zz, PE_xy, PE_yz, PE_xz  (plastic strain)

    Residual r = σ - C : (ε - ε^p).
    Voigt-Hooke matrix built from Lamé constants (λ, μ) derived from (E, ν).

    Returns MSE of the residual across all nodes. Returns 0 if fewer than 21 channels.
    """
    if pred.shape[-1] < 21:
        return pred.new_zeros(())

    lam = (E_modulus * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E_modulus / (2.0 * (1.0 + nu))

    # Voigt-form isotropic C  [6, 6]  (engineering shear strain assumed)
    C = torch.zeros(6, 6, device=pred.device, dtype=pred.dtype)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu

    sigma_p = pred[:, 3:9]
    eps_p   = pred[:, 9:15]
    epsp_p  = pred[:, 15:21]

    residual = sigma_p - (eps_p - epsp_p) @ C.T
    return (residual ** 2).mean()


def constitutive_loss_sliced(sspe_pred, sspe_true, E_modulus=70e3, nu=0.33):
    """
    Same constitutive residual as constitutive_loss, but operates directly on
    an 18-channel [S(6), LE(6), PE(6)] slice — used when output_mode='sspe'
    (no displacements in the output).
    """
    if sspe_pred.shape[-1] < 18:
        return sspe_pred.new_zeros(())

    lam = (E_modulus * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu  = E_modulus / (2.0 * (1.0 + nu))

    C = torch.zeros(6, 6, device=sspe_pred.device, dtype=sspe_pred.dtype)
    C[:3, :3] = lam
    C[0, 0] = C[1, 1] = C[2, 2] = lam + 2.0 * mu
    C[3, 3] = C[4, 4] = C[5, 5] = mu

    sigma_p = sspe_pred[:, 0:6]
    eps_p   = sspe_pred[:, 6:12]
    epsp_p  = sspe_pred[:, 12:18]

    residual = sigma_p - (eps_p - epsp_p) @ C.T
    return (residual ** 2).mean()


@torch.no_grad()
def precompute_lambda_E(model, loader, criterion, device, disp_slice, offset=1.0, max_batches=None):
    """
    Compute lambda_E ONCE before training using the user's compute_lambda scheme.
    Averages over one pass of the training loader (or up to max_batches batches).

    Uses the untrained model's predictions to estimate the order-of-magnitude
    ratio between displacement MSE and displacement-gradient MSE, then returns
    a fixed lambda_E to hold throughout training.

    Returns: (lambda_E_scalar, diagnostic_dict)
    """
    if disp_slice is None:
        return 0.0, {"skipped": True, "reason": "no displacement slice"}

    model.eval()
    lams = []
    l_disps, l_strains = [], []

    for i, data in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        data = data.to(device)
        pred = model(data)
        p_d = pred[:, disp_slice]
        t_d = data.y[:, disp_slice]
        pos = data.pos if hasattr(data, "pos") and data.pos is not None else data.x[:, :3]

        lam_b = compute_lambda_adaptive(p_d, t_d, pos, data.edge_index, criterion, offset=offset)

        # Also record raw magnitudes for diagnostics
        l_d = criterion(p_d, t_d)
        pred_strain = compute_strain(p_d, pos, data.edge_index)
        true_strain = compute_strain(t_d, pos, data.edge_index)
        l_s = criterion(pred_strain, true_strain) if pred_strain.numel() > 0 else pred.new_zeros(())

        lams.append(float(lam_b.item()))
        l_disps.append(float(l_d.item()))
        l_strains.append(float(l_s.item()))

    if not lams:
        return 0.0, {"skipped": True, "reason": "empty loader"}

    lam_final = float(np.mean(lams))
    diag = {
        "lam_batches":       lams,
        "lam_mean":          lam_final,
        "lam_std":           float(np.std(lams)),
        "lam_median":        float(np.median(lams)),
        "l_disp_mean":       float(np.mean(l_disps)),
        "l_strain_mean":     float(np.mean(l_strains)),
        "offset_used":       offset,
        "n_batches_sampled": len(lams),
    }
    return lam_final, diag


# =============================================================================
# DATA LOADING (mirrors user's loader)
# =============================================================================

def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', os.path.basename(s))]


def load_data_list(root_dir, recursive=False):
    """
    Load all *.pt PyG Data files from a directory (natural-sorted).
    Returns list of Data objects.
    """
    if os.path.isdir(root_dir):
        pattern = "**/*.pt" if recursive else "*.pt"
        files = glob.glob(os.path.join(root_dir, pattern), recursive=recursive)
    else:
        files = glob.glob(root_dir, recursive=True)
    files = sorted(files, key=_natural_key)

    data_list = []
    for fp in tqdm(files, desc="Loading .pt"):
        obj = torch.load(fp, map_location="cpu", weights_only=False)
        if isinstance(obj, Data):
            data_list.append(obj)
        elif isinstance(obj, (list, tuple)) and all(isinstance(x, Data) for x in obj):
            data_list.extend(obj)
        elif isinstance(obj, dict) and "data" in obj:
            data_list.extend(obj["data"])
    return data_list


# =============================================================================
# SPLIT + STATS
# =============================================================================

def compute_stats(data_list):
    """Dataset-wide min/max/mean/std for inputs and outputs."""
    xs = torch.cat([d.x for d in data_list], dim=0)
    ys = torch.cat([d.y.reshape(-1, d.y.shape[-1]) for d in data_list], dim=0)
    stats = {
        "input_stats":  {"min": xs.min(0).values, "max": xs.max(0).values,
                         "mean": xs.mean(0),      "std": xs.std(0)},
        "output_stats": {"min": ys.min(0).values, "max": ys.max(0).values,
                         "mean": ys.mean(0),      "std": ys.std(0)},
    }
    return stats


def denorm_output(y_norm, stats, mode="minmax"):
    """Inverse of the normalization used on d.y."""
    y = y_norm.detach().clone()
    if mode == "minmax":
        ymin = stats["output_stats"]["min"].to(y.device, y.dtype)
        ymax = stats["output_stats"]["max"].to(y.device, y.dtype)
        return y * (ymax[:y.shape[-1]] - ymin[:y.shape[-1]]) + ymin[:y.shape[-1]]
    if mode == "standard":
        mu = stats["output_stats"]["mean"].to(y.device, y.dtype)
        sd = stats["output_stats"]["std"].to(y.device, y.dtype)
        return y * sd[:y.shape[-1]] + mu[:y.shape[-1]]
    return y


# =============================================================================
# TRAIN / EVAL SCAFFOLD SHARED BY ALL SCRIPTS
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device,
                    lambda_E=0.0, lambda_M=0.0,
                    clip=1.0, epoch=None, total_epochs=None,
                    show_progress=True,
                    disp_slice=None, sspe_slice=None):
    """
    disp_slice : slice or None. Which channels of pred/data.y are displacements.
                 If None, L_E is disabled (no displacement in outputs).
    sspe_slice : slice or None. Which channels are stress/strain/PE for L_M.
                 If None or covers < 18 channels, L_M is disabled.
    """
    model.train()
    total = 0.0
    total_disp = 0.0
    total_E = 0.0
    total_M = 0.0
    n = 0

    desc = f"ep{epoch:>4d}/{total_epochs}" if (epoch is not None and total_epochs is not None) else "train"
    iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True) if show_progress else loader

    # L_E requires a displacement slice; skip it silently otherwise (sspe mode)
    do_E = (lambda_E > 0) and (disp_slice is not None)
    do_M = (lambda_M > 0) and (sspe_slice is not None)

    for data in iterator:
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data)                                       # normalized space

        # main supervised loss on FULL prediction tensor
        l_disp = criterion(pred, data.y)
        loss = l_disp

        if do_E:
            pos = data.pos if hasattr(data, "pos") and data.pos is not None else data.x[:, :3]
            l_E = displacement_gradient_loss(
                pred[:, disp_slice], data.y[:, disp_slice], pos, data.edge_index
            )
            loss = loss + lambda_E * l_E
            total_E += float(l_E.item())

        if do_M:
            # constitutive_loss needs the full 18-channel S/LE/PE block; slice out
            l_M = constitutive_loss_sliced(pred[:, sspe_slice], data.y[:, sspe_slice])
            loss = loss + lambda_M * l_M
            total_M += float(l_M.item())

        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total += float(loss.item())
        total_disp += float(l_disp.item())
        n += 1

        if show_progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{loss.item():.3e}",
                disp=f"{l_disp.item():.3e}",
            )

    return {
        "total":     total / max(1, n),
        "l_disp":    total_disp / max(1, n),
        "l_E":       total_E / max(1, n) if do_E else 0.0,
        "l_M":       total_M / max(1, n) if do_M else 0.0,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, stats, norm_mode="minmax"):
    """
    Run inference; return
        - list of per-graph physical predictions (list of [N_i, C])
        - list of per-graph physical ground truths (list of [N_i, C])
        - mean normalized-space MSE (scalar)
    """
    model.eval()
    preds_phys, trues_phys, mse_norm_list = [], [], []
    for data in loader:
        data = data.to(device)
        pred = model(data)
        mse_norm_list.append(float(criterion(pred, data.y).item()))

        # split back into individual graphs using data.batch
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            for gid in range(int(batch.max().item()) + 1):
                mask = (batch == gid)
                p_phys = denorm_output(pred[mask], stats, mode=norm_mode).cpu()
                t_phys = denorm_output(data.y[mask], stats, mode=norm_mode).cpu()
                preds_phys.append(p_phys)
                trues_phys.append(t_phys)
        else:
            preds_phys.append(denorm_output(pred, stats, mode=norm_mode).cpu())
            trues_phys.append(denorm_output(data.y, stats, mode=norm_mode).cpu())

    return preds_phys, trues_phys, float(np.mean(mse_norm_list))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
