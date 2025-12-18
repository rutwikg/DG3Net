# Script to predict from the saved model and write to VTU
import os
import re
import glob
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from natsort import natsorted
import pyvista as pv

# ============================================================
# 1) Model: (exactly as in training)
# ============================================================

# Insert model here by replacing the placeholder below
class MeshGATv2NodeRegression(nn.Module):
    def __init__(self):
        super(MeshGATv2NodeRegression, self).__init__()
        pass
    
    def forward(self, x):
        return x


def load_model_and_metadata(model_path: str,  device: torch.device, meta_path: str | None = None):
    """
    Load model + metadata from a checkpoint.

    Supports:
      1) New-style dict:
         {
           "model": <nn.Module or state_dict>,
           "metadata": <dict or bundle['metadata']>,
           "Training Stats": {
               'hidden_dim': ...,
               'output_dim': ...,
               'FinalLayerActivation': ...,
               'nhead': ...,
               'Test_indices': ...,
               'Train_indices': ...
           }
         }

      2) Old-style:
         - checkpoint is a full nn.Module
         - or a plain state_dict, with metadata supplied via --meta.
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    model = None
    meta = None
    training_stats = {}

    # ---------- Case 1: new-style dict checkpoint ----------
    if isinstance(ckpt, dict) and (
        "model" in ckpt or "metadata" in ckpt or "Training Stats" in ckpt
    ):  
        print('New style checkpoint detected.')
        # Training stats (optional)
        if "Training Stats" in ckpt and isinstance(ckpt["Training Stats"], dict):
            training_stats = ckpt["Training Stats"]

        # ---- Model ----
        if "model" in ckpt:
            mobj = ckpt["model"]
            if isinstance(mobj, nn.Module):
                model = mobj.to(device)
                print("[INFO] Loaded nn.Module from 'model' key in checkpoint.")
            else:
                # assume it's a state_dict
                hidden_dim = training_stats.get("hidden_dim", 64)
                output_dim = training_stats.get("output_dim", 3)
                heads      = training_stats.get("nhead", 4)
                print(f"[INFO] Rebuilding MeshGATv2NodeRegression from state_dict "
                      f"(hidden_dim={hidden_dim}, output_dim={output_dim}, heads={heads}).")
                model = MeshGATv2NodeRegression(
                    input_dim=4,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    heads=heads
                ).to(device)
                model.load_state_dict(mobj)
        else:
            # Fall through to legacy handling below if no 'model' key
            pass

        # ---- Metadata ----
        if "metadata" in ckpt:
            meta_obj = ckpt["metadata"]
            if isinstance(meta_obj, dict) and "metadata" in meta_obj:
                meta = meta_obj["metadata"]
            else:
                meta = meta_obj

    # ---------- Case 2: legacy checkpoint (full module or state_dict) ----------
    if model is None:
        print('Legacy checkpoint detected.')
        if isinstance(ckpt, nn.Module):
            model = ckpt.to(device)
            print("[INFO] Loaded full nn.Module checkpoint (legacy style).")
        else:
            # Plain state_dict; use Training Stats if available, else defaults
            # input_dim =  training_stats.get("input_dim", 4)
            hidden_dim = training_stats.get("hidden_dim", 64)
            output_dim = training_stats.get("output_dim", 3)
            heads      = training_stats.get("nhead", 4)
            print(f"[INFO] Loaded state_dict (legacy); rebuilding MeshGATv2NodeRegression "
                  f"(hidden_dim={hidden_dim}, output_dim={output_dim}, heads={heads}).")
            model = MeshGATv2NodeRegression(
                input_dim=4,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                heads=heads
            ).to(device)
            model.load_state_dict(ckpt)

    # ---------- Metadata fallback ----------
    if meta is None:
        print('No metadata found in checkpoint, trying --meta path.')
        if meta_path is None:
            raise ValueError(
                "No metadata found inside the checkpoint and no --meta file provided.\n"
                "Either save 'metadata' in the checkpoint dict, or pass --meta <path>."
            )
        meta_obj = torch.load(meta_path, map_location="cpu", weights_only=False)
        if isinstance(meta_obj, dict) and "metadata" in meta_obj:
            meta = meta_obj["metadata"]
        else:
            meta = meta_obj

    # ---------- Metadata device alignment ----------
    # meta = move_meta_stats_to_device(meta, device)

    # Some logging
    norm_mode = meta.get("norm_mode", "standard") if isinstance(meta, dict) else "unknown"
    print(f"[INFO] norm_mode={norm_mode}")
    if training_stats:
        print(f"[INFO] Training Stats found: {training_stats}")

    return model, meta, training_stats

def move_meta_stats_to_device(meta: dict, device: torch.device):
    """
    Move normalization statistics in `meta` (input_stats/output_stats)
    to the given device (cpu or cuda).
    """
    if not isinstance(meta, dict):
        return meta

    for k in ["input_stats", "output_stats"]:
        if k in meta and isinstance(meta[k], dict):
            stat_dict = meta[k]
            for name in ["min", "max", "mean", "std"]:
                t = stat_dict.get(name, None)
                if torch.is_tensor(t):
                    stat_dict[name] = t.to(device)
    return meta


# ============================================================
# 2) Normalization / denormalization (same as training)
# ============================================================

def normalize_inputs_only(
    data_list,
    stats,
    mode: str = "minmax",
    eps: float = 1e-9,
):
    """
    Normalize ONLY x (inputs) using stats['input_stats'].
    This avoids any mismatch with y-dimension at inference.
    """
    valid_modes = ["minmax", "standard"]
    if mode not in valid_modes:
        raise ValueError(
            f"[normalize_inputs_only] Invalid mode '{mode}'. Allowed: {valid_modes}"
        )

    s_in = stats.get("input_stats", None)
    out = []
    for d in data_list:
        d2 = d.clone()

        if s_in is not None and hasattr(d2, "x") and d2.x is not None:
            arr = d2.x.float()
            if mode == "minmax":
                arr = (arr - s_in["min"]) / (s_in["max"] - s_in["min"] + eps)
            else:  # "standard"
                arr = (arr - s_in["mean"]) / s_in["std"]
            d2.x = arr

        # DO NOT TOUCH d2.y HERE
        out.append(d2)

    return out

# =========== Geometric Normalization Helpers ===========

# ------------------------- Geometry aware Normalization ---------------------------------
# ===== Add once: helpers =====

import torch
from typing import Iterable, Dict
COORD_IDX = (0, 1, 2)

def _extract_coords(x: torch.Tensor, coord_idx=COORD_IDX):
    return x[:, torch.tensor(coord_idx)]

def _replace_coords(x: torch.Tensor, new_coords: torch.Tensor, coord_idx=COORD_IDX):
    x2 = x.clone()
    x2[:, torch.tensor(coord_idx)] = new_coords
    return x2

def normalize_coords_per_graph(data, coord_idx=COORD_IDX, eps=1e-12, store=True):
    coords = _extract_coords(data.x, coord_idx)
    c = coords.mean(dim=0, keepdim=True)
    centered = coords - c
    bb_min, bb_max = centered.min(0).values, centered.max(0).values
    L = torch.norm(bb_max - bb_min).clamp_min(eps)
    coords_n = centered / L
    data.x = _replace_coords(data.x, coords_n, coord_idx)
    if store:
        data.geom_centroid = c.squeeze(0)
        data.geom_L = L
    return data

def normalize_coords_many(data_list: Iterable, coord_idx=COORD_IDX):
    return [normalize_coords_per_graph(d, coord_idx=coord_idx) for d in data_list]

def _build_feature_masks(x_dim: int, coord_idx=COORD_IDX):
    mask = torch.ones(x_dim, dtype=torch.bool)
    mask[torch.tensor(coord_idx)] = False
    return mask

def compute_io_stats_excluding_coords(data_list: Iterable, coord_idx=COORD_IDX):
    x_vals, y_vals = [], []
    for d in data_list:
        if hasattr(d, "x") and d.x is not None: x_vals.append(d.x.detach().cpu().float())
        if hasattr(d, "y") and d.y is not None: y_vals.append(d.y.detach().cpu().float())

    stats = {"input_stats": None, "output_stats": None}
    if x_vals:
        X = torch.cat(x_vals, dim=0)
        mask_non_geom = _build_feature_masks(X.shape[1], coord_idx)
        Xn = X[:, mask_non_geom]
        stats["input_stats"] = {
            "min":  Xn.min(0).values,
            "max":  Xn.max(0).values,
            "mean": Xn.mean(0),
            "std":  Xn.std(0, unbiased=False).clamp_min(1e-9),
            "mask_non_geom": mask_non_geom,
        }
    if y_vals:
        Y = torch.cat(y_vals, dim=0)
        stats["output_stats"] = {
            "min":  Y.min(0).values,
            "max":  Y.max(0).values,
            "mean": Y.mean(0),
            "std":  Y.std(0, unbiased=False).clamp_min(1e-9),
        }
    return stats

def normalize_nongeom_features(data_list: Iterable, stats: Dict, mode="minmax", eps=1e-9, coord_idx=COORD_IDX):

    valid_modes = ["minmax", "standard"]
    if mode not in valid_modes:
        raise ValueError(
            f"[normalize_with_io_stats] Invalid mode '{mode}'. Allowed: {valid_modes}"
        )
    
    print('Using normalize_nongeom_features with mode:', mode)
    out = []
    s_in = stats.get("input_stats", None)
    s_out = stats.get("output_stats", None)
    for d in data_list:
        d2 = d.clone()
        # Inputs (only non-geometry dims)
        if s_in and hasattr(d2, "x") and d2.x is not None:
            x = d2.x.clone()
            mask = s_in["mask_non_geom"]
            if mode == "minmax":
                x[:, mask] = (x[:, mask] - s_in["min"]) / (s_in["max"] - s_in["min"] + eps)
            else:
                x[:, mask] = (x[:, mask] - s_in["mean"]) / s_in["std"]
            d2.x = x
        # Outputs (all dims)
        if s_out and hasattr(d2, "y") and d2.y is not None:
            y = d2.y.clone()
            if mode == "minmax":
                y = (y - s_out["min"]) / (s_out["max"] - s_out["min"] + eps)
            else:
                y = (y - s_out["mean"]) / s_out["std"]
            d2.y = y
        out.append(d2)
    return out

def ensure_edge_attr_distance_direction(data, coord_idx=COORD_IDX):
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return data
    i, j = data.edge_index
    rel = _extract_coords(data.x, coord_idx)[j] - _extract_coords(data.x, coord_idx)[i]
    dist = rel.norm(dim=1, keepdim=True)
    dirn = rel / (dist + 1e-12)
    new_edge_attr = torch.cat([dist, dirn], dim=1)  # [E,4]
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=1)
    else:
        data.edge_attr = new_edge_attr
    return data


# ============ Standard Normalization Helpers ============

def normalize_with_io_stats(
    data_list,
    stats,
    mode: str = "minmax",
    eps: float = 1e-9,
):
    """
    Same logic you used in training: global normalization of x and y.
    stats should contain "input_stats" and "output_stats".
    """
    valid_modes = ["minmax", "standard"]
    if mode not in valid_modes:
        raise ValueError(
            f"[normalize_with_io_stats] Invalid mode '{mode}'. Allowed: {valid_modes}"
        )

    s_in = stats.get("input_stats", None)
    s_out = stats.get("output_stats", None)

    out = []
    for d in data_list:
        d2 = d.clone()

        # Inputs
        if s_in is not None and hasattr(d2, "x") and d2.x is not None:
            arr = d2.x.float()
            if mode == "minmax":
                arr = (arr - s_in["min"]) / (s_in["max"] - s_in["min"] + eps)
            else:
                arr = (arr - s_in["mean"]) / s_in["std"]
            d2.x = arr

        # Outputs
        if s_out is not None and hasattr(d2, "y") and d2.y is not None:
            arr = d2.y.float()
            if mode == "minmax":
                arr = (arr - s_out["min"]) / (s_out["max"] - s_out["min"] + eps)
            else:
                arr = (arr - s_out["mean"]) / s_out["std"]
            d2.y = arr

        out.append(d2)

    return out


def denorm_predictions(predictions, meta, key: str = "y"):
    """
    Denormalize a list of predictions using stats from meta (not full bundle).
    This mirrors your training helper, but takes `meta` directly.

    predictions: list of [N, F] tensors
    meta: dict with "input_stats"/"output_stats" and "norm_mode"
    key: "x" or "y" (we use "y" here)
    """
    mode = meta.get("norm_mode", None)
    if mode is None:
        print("[WARN] No 'norm_mode' in metadata; returning unchanged predictions.")
        return predictions

    stats = meta["input_stats"] if key == "x" else meta["output_stats"]

    out_list = []
    for arr in predictions:
        arr = arr.clone().detach()
        F = arr.shape[-1]

        if mode == "minmax":
            lo = stats["min"][:F].to(arr.device, arr.dtype)
            hi = stats["max"][:F].to(arr.device, arr.dtype)
            arr = arr * (hi - lo) + lo
        elif mode == "standard":
            mu = stats["mean"][:F].to(arr.device, arr.dtype)
            sd = stats["std"][:F].to(arr.device, arr.dtype)
            arr = arr * sd + mu

        out_list.append(arr)

    return out_list

def denorm_predictions_geom(predictions, meta, key="y"):
    """
    Denormalize a list of prediction tensors using geometry-aware stats.

    - For key == "y": uses output_stats (all dims).
    - For key == "x": uses input_stats *without* touching coordinates
      (only non-geometry dims, based on mask_non_geom).
    """
    mode = meta.get("norm_mode", None)
    
    if mode is None:
        print("[WARN] No normalization mode in metadata, returning unchanged.")
        return predictions

    stats = meta["input_stats"] if key == "x" else meta["output_stats"]
    if stats is None:
        print("[WARN] No stats found for key", key)
        return predictions

    out_list = []
    for arr in predictions:
        arr = arr.clone().detach()
        F = arr.shape[-1]

        if key == "y":
            # simple global denorm over all output features
            if mode == "minmax":
                lo = stats["min"][:F].to(arr.device, arr.dtype)
                hi = stats["max"][:F].to(arr.device, arr.dtype)
                arr = arr * (hi - lo) + lo
            else:  # "standard"
                mu = stats["mean"][:F].to(arr.device, arr.dtype)
                sd = stats["std"][:F].to(arr.device, arr.dtype)
                arr = arr * sd + mu

        else:  # key == "x" -> only non-geom dims are normalized
            mask = stats["mask_non_geom"][:F].to(arr.device)
            arr_out = arr.clone()
            if mode == "minmax":
                lo = stats["min"].to(arr.device, arr.dtype)
                hi = stats["max"].to(arr.device, arr.dtype)
                arr_out[:, mask] = arr[:, mask] * (hi - lo) + lo
            else:
                mu = stats["mean"].to(arr.device, arr.dtype)
                sd = stats["std"].to(arr.device, arr.dtype)
                arr_out[:, mask] = arr[:, mask] * sd + mu
            arr = arr_out

        out_list.append(arr)

    return out_list

# ============================================================
# 3) VTU/PVD helpers
# ============================================================

def find_pvd(run_dir: str) -> str:
    """Locate a .pvd file inside a Design_{ID} directory."""
    cands = glob.glob(os.path.join(run_dir, "*.pvd"))
    if not cands:
        raise FileNotFoundError(f"No .pvd found in {run_dir}")
    return cands[0]


def get_last_vtu_from_pvd(pvd_path: str) -> str:
    """Parse the PVD and return filename of the last time step."""
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    datasets = []
    for ds in root.iter("DataSet"):
        t = float(ds.attrib.get("timestep", "0.0"))
        f = ds.attrib["file"]
        datasets.append((t, f))
    if not datasets:
        raise RuntimeError(f"No DataSet entries in {pvd_path}")
    datasets.sort(key=lambda x: x[0])
    _, last_file = datasets[-1]
    return last_file  # relative to pvd folder


def write_last_step_vtu(
    design_dir_name: str,
    vtu_root: str,
    out_root: str,
    pred_disp,
    act_disp,
    percent_error,
    pred_comp,                                        # [N, 3], denormalized
    field_name: str = "U_vec_predicted",
):
    """
    For a given Design_* folder:
      - Find the last VTU in Test_VTU/Design_*
      - Attach predicted displacement as field_name
      - Save as Pred_Test/Design_*/<same_vtu_name>
    """
    design_dir = os.path.join(vtu_root, design_dir_name)
    if not os.path.isdir(design_dir):
        raise FileNotFoundError(f"Design directory not found: {design_dir}")

    pvd_path = find_pvd(design_dir)
    last_rel = get_last_vtu_from_pvd(pvd_path)
    last_vtu_path = os.path.join(design_dir, last_rel)

    mesh = pv.read(last_vtu_path)
    n_pts = mesh.points.shape[0]

    pred_disp = pred_disp.detach().cpu().numpy()
    if pred_disp.ndim == 3:
        pred_disp = pred_disp.reshape(pred_disp.shape[-2], pred_disp.shape[-1])
    if pred_disp.shape[1] > 3:
        pred_disp = pred_disp[:, :3]

    if pred_disp.shape != (n_pts, 3):
        raise ValueError(
            f"[{design_dir_name}] node mismatch: pred {pred_disp.shape} vs VTU points {n_pts}"
        )

    out_dir = os.path.join(out_root, design_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    mesh[field_name] = pred_disp
    mesh["Actual displacement"] = act_disp.detach().cpu().numpy()
    mesh['Error (%)'] = percent_error.detach().cpu().numpy()

    out_vtu = os.path.join(out_dir, os.path.basename(last_vtu_path))
    mesh.save(out_vtu)

    print(f"[OK] {design_dir_name}: wrote {out_vtu}")


# ============================================================
# 4) Main: load graphs → normalize → predict → denorm → VTU
# ============================================================

def run_prediction_field(
    graphs_root: str,
    model_path: str,
    meta_path: str,
    vtu_root: str,
    out_root: str,
    device: torch.device,
    geo_norm: bool = False,
):
    # ---------- load model + metadata (new or legacy) ----------
    model, meta, training_stats = load_model_and_metadata(model_path, device, meta_path)
    norm_mode = meta.get("norm_mode", "standard")
    model.eval()


    # ---------- list DG3 graphs and VTU design dirs ----------
    dg3_files = [f for f in natsorted(os.listdir(graphs_root)) if f.endswith(".dg3")]
    design_dirs = [d for d in natsorted(os.listdir(vtu_root))
                   if os.path.isdir(os.path.join(vtu_root, d))]

    if len(dg3_files) != len(design_dirs):
        print(f"[WARN] #graphs={len(dg3_files)} != #design_dirs={len(design_dirs)} "
              f"(will only process min of both).")

    n = min(len(dg3_files), len(design_dirs))
    print(f"[INFO] Processing {n} pairs of (graph, design_dir)")

    crit = nn.MSELoss()

    for i in range(n):
        dg3_name = dg3_files[i]
        design_dir = design_dirs[i]
        dg3_path = os.path.join(graphs_root, dg3_name)

        print(f"\n[PRED] Graph={dg3_name} | Design={design_dir}")
        # ----- load raw graph -----
        data_raw: Data = torch.load(dg3_path, map_location="cpu", weights_only=False)

        if data_raw.x.shape[1] < 4:
            raise RuntimeError(f"x has {data_norm.x.shape[1]} features, expected >=4 for concat.")
        
        x4 = torch.cat((data_raw.x[:, :3], data_raw.x[:, -1:]), dim=1)
        data_raw.x = x4
        y3 = data_raw.y[:,-1,:3]
        data_raw.y = y3

        if geo_norm:
            # ----- geometry-aware normalization -----
            print("Using GEO-Aware normalization.")
            COORD_IDX = (0, 1, 2)
            data_geom = normalize_coords_many([data_raw],coord_idx=COORD_IDX)[0]
            data_geom = [ensure_edge_attr_distance_direction(d) for d in ([data_geom])][0]
            data_norm = normalize_nongeom_features([data_geom], stats=meta, mode=norm_mode[4:],coord_idx=COORD_IDX)[0]
        else:
            # ----- normalize x,y (global stats) -----
            print("Using Conventional normalization.")
            data_norm = normalize_with_io_stats([data_raw], stats=meta, mode=norm_mode)[0]

        # ----- match training preprocessing -----
        # x: concat coords[:3] and last feature only -> 4-dim input
        

        

        # y: last time step and first 3 components if 3D tensor; else first 3 columns
        if hasattr(data_norm, "y") and data_norm.y is not None:
            if data_norm.y.dim() == 3:
                print('Reducing TH to Field (last time step, first 3 components).')
                data_norm.y = data_norm.y[:, -1, :3]
            elif data_norm.y.dim() == 2 and data_norm.y.shape[1] >= 3:
                data_norm.y = data_norm.y[:, :3]

        # ----- prediction -----
        data_dev = data_norm.to(device)
        with torch.no_grad():
            pred_norm = model(data_dev)     # [N, 3] in normalized space

        # Optional: compute normalized-space MSE if ground truth present
        if hasattr(data_norm, "y") and data_norm.y is not None:
            gt = data_norm.y.to(device)
            if gt.shape == pred_norm.shape:
                loss = crit(pred_norm, gt)
                print(f"   MSE (normalized): {loss.item():.6e}")

        # ----- denormalize predictions to physical units -----
        if geo_norm:
            pred_denorm_list = denorm_predictions_geom([pred_norm.cpu()], meta, key="y")
            pred_denorm = pred_denorm_list[0]   # [N, 3]
            act_denorm_list = denorm_predictions_geom([data_norm.y.cpu()], meta, key="y")
            act_denorm = act_denorm_list[0]     # [N, 3]
            percent_error = torch.norm(pred_denorm - act_denorm) / (torch.norm(act_denorm, dim=1) + 1e-9) * 100.0

        else:
            pred_denorm_list = denorm_predictions([pred_norm.cpu()], meta, key="y")
            pred_denorm = pred_denorm_list[0]   # [N, 3]
            act_denorm_list = denorm_predictions([data_norm.y.cpu()], meta, key="y")
            act_denorm = act_denorm_list[0]     # [N, 3]
            percent_error = torch.norm(pred_denorm - act_denorm) / (torch.norm(act_denorm, dim=1) + 1e-9) * 100.0


        # ----- write to last-step VTU -----
        write_last_step_vtu(
            design_dir_name=design_dir,
            vtu_root=vtu_root,
            out_root=out_root,
            pred_disp=pred_denorm,
            act_disp=act_denorm,
            percent_error=percent_error,
            pred_comp = predd_denorm,
            field_name="Predicted Displacement",
        )

    print(f"\n[DONE] All predictions written to: {out_root}")


# ============================================================
# 5) CLI
# ============================================================

if __name__ == "__main__":

    import argparse

    ap = argparse.ArgumentParser(
        description="Predict final-step displacements using MeshGATv2NodeRegression and write them into last VTU."
    )
    ap.add_argument("--graphs-root", required=True,
                    help="Directory with test graph .pt files (PyG Data).")
    ap.add_argument("--model", required=True,
                    help="Trained model .pt (full model or state_dict).")
    ap.add_argument("--meta",required=False,default=None,
    help=(
        "Optional meta file (.pt). "
        "If the model checkpoint contains 'metadata', this is ignored. "
        "If not, this must point to a bundle or metadata dict."
    ),)

    ap.add_argument("--vtk-root", required=True,
                    help="Root with Test_VTU (Design_* subfolders with PVD+VTU).")
    ap.add_argument("--out-root", required=True,
                    help="Output root, e.g. Pred_Test.")
    ap.add_argument("--geo", action="store_true",
                    help="Enables Geo Norm when true.")
    
    ap.add_argument("--device",choices=["cpu", "cuda"],default="cpu",
                    help="Device to run normalization + model on (default: cpu).")

    args = ap.parse_args()

    # Resolve device from flag
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    run_prediction_field(
        graphs_root=args.graphs_root,
        model_path=args.model,
        meta_path=args.meta,
        vtu_root=args.vtk_root,
        out_root=args.out_root,
        device=device,
        geo_norm=args.geo,
        
    )


