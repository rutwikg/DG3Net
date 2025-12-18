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

# geo_aware_norm.py
# Computes geometry-aware normalization from the DG3Net paper

COORD_IDX = (0, 1, 2)

import torch
from typing import Iterable, Dict

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
