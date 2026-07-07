"""
compute_baseline_metrics.py
===========================
Reads the trained baseline checkpoints from LiteratureCompare.ipynb and
DG3Net, runs inference on the same held-out 3D test set, and reports the five
reviewer-required metrics per model + per-sample data for scatter/error plots.

Assumes:
    - baseline models were saved via   torch.save(full_model_object, path)
      i.e. loadable with torch.load(..., weights_only=False)
    - all baselines share the SAME normalization bundle (min-max stats)
    - EA-GNN test set is data_mm_ea, everyone else uses data_mm
    - held-out test set is the same 45 samples the paper reports

Output:
    <out>/baseline_metrics.json     aggregate + per-sample tables
    <out>/baseline_scatter.pkl      per-sample true-vs-pred peak displacement
    <out>/baseline_hist_data.pkl    per-sample relative-L2 errors for histograms

Usage:
    python compute_baseline_metrics.py \
        --root      /path/to/graphs \
        --meta      /path/to/meta_mm.pt \
        --ckpts_dir /path/to/trained \
        --dg3_ckpt  /path/to/DG3Net_trained.pt \
        --out       ./baseline_report \
        --gpu       0 \
        --output_dim 3
"""
import argparse, os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from metrics import (
    evaluate_over_set,
    stratified_train_val_test_split,
    format_aggregate_table,
)
from common import (
    load_data_list,
    compute_stats,
    apply_minmax,
    denorm_output,
)
# IMPORTANT: importing baseline_models registers the class definitions in the
# global namespace so torch.load(..., weights_only=False) can unpickle the
# full-model .pt files saved by train_baselines.py / LiteratureCompare.ipynb.
import baseline_models   # noqa: F401
from baseline_models import (
    MeshGraphNet, XMeshGraphNet, EAGNN, GraphUNet,
    compute_edge_attr_full as compute_edge_attr_full_ext,
    add_augmented_edges as add_augmented_edges_ext,
)


# ----------- utilities ------------------------------------------------------

def slim_x_y(data_list, input_dim, output_dim):
    for d in data_list:
        if d.y.ndim == 3:
            d.y = d.y[:, -1, :output_dim]
        else:
            d.y = d.y[:, :output_dim]
        if d.x.shape[1] > input_dim:
            d.x = torch.cat([d.x[:, :3], d.x[:, -1:]], dim=1)
    return data_list


def compute_edge_attr_full(data_list):
    """8-dim edge attr matching user's compute_edge_attr(mode='full')."""
    out = []
    for d in data_list:
        d2 = d.clone()
        src, dst = d2.edge_index
        pos  = d2.x[:, :3]
        rel  = pos[dst] - pos[src]
        length = rel.norm(dim=1, keepdim=True).clamp_min(1e-12)
        norm_dir = rel / length
        thick     = d2.x[:, 3:4]
        rel_thick = thick[dst] - thick[src]
        d2.edge_attr = torch.cat([length, rel, rel_thick, norm_dir], dim=1)  # [E, 8]
        out.append(d2)
    return out


def add_augmented_edges_for_eagnn(data_list, k_aug=16):
    """Mirror of user's add_augmented_edges (kNN in coord space)."""
    from torch_geometric.utils import to_undirected

    def knn_pytorch(pos, k):
        diff  = pos.unsqueeze(0) - pos.unsqueeze(1)
        d2    = (diff ** 2).sum(dim=-1); d2.fill_diagonal_(float('inf'))
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
        d2.edge_attr_aug  = torch.cat([dist_a, dir_a], dim=1)  # [E_aug, 4]
        out.append(d2)
    return out


@torch.no_grad()
def infer_and_collect(model, loader, device, stats, norm_mode="minmax"):
    """
    Run inference on `loader` and return per-sample physical predictions
    and ground truths.
    """
    model.eval()
    preds, trues = [], []
    for data in tqdm(loader, desc="Inference"):
        data = data.to(device)
        pred = model(data)                          # normalized space
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            for gid in range(int(batch.max().item()) + 1):
                mask = (batch == gid)
                preds.append(denorm_output(pred[mask], stats, mode=norm_mode).cpu())
                trues.append(denorm_output(data.y[mask], stats, mode=norm_mode).cpu())
        else:
            preds.append(denorm_output(pred, stats, mode=norm_mode).cpu())
            trues.append(denorm_output(data.y, stats, mode=norm_mode).cpu())
    return preds, trues


# ---- Main -----------------------------------------------------------------

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--root",       type=str, required=True)
    p.add_argument("--meta",       type=str, default=None,
                   help="Optional path to meta_mm.pt (pre-computed min-max stats). "
                        "If missing, stats are computed from --root.")
    p.add_argument("--ckpts_dir",  type=str, required=True,
                   help="Directory containing MGN_trained.pt, XMGN_trained.pt, "
                        "EAGNN_trained.pt, GraphUNet_trained.pt")
    p.add_argument("--gatv2_ckpt", type=str, default=None,
                   help="Optional GATv2 checkpoint (from FieldTransformer_Geo.ipynb)")
    p.add_argument("--dg3_ckpt",   type=str, default=None,
                   help="Optional DG3Net (FEMAT) checkpoint")
    p.add_argument("--out",         type=str, default="./baseline_report")
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--input_dim",   type=int, default=4)
    p.add_argument("--output_dim",  type=int, default=3)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--stratify_key", type=str, default=None)
    p.add_argument("--models",      type=str, nargs="+",
                   default=["mgn", "xmgn", "eagnn", "gunet", "gatv2", "dg3net"],
                   help="Subset of {mgn, xmgn, eagnn, gunet, gatv2, dg3net} to evaluate")
    return p.parse_args()


def get_stratify_labels(data_list, key):
    if key is None:
        return np.zeros(len(data_list), dtype=int)
    labels = []
    for d in data_list:
        v = getattr(d, key, 0)
        try:
            labels.append(int(v) if not torch.is_tensor(v) else int(v.item()))
        except Exception:
            labels.append(0)
    return np.array(labels)


def main():
    args = parse()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    data_list = load_data_list(args.root)
    data_list = slim_x_y(data_list, args.input_dim, args.output_dim)

    if args.meta and os.path.isfile(args.meta):
        print(f"Loading pre-computed stats from {args.meta}")
        meta = torch.load(args.meta, map_location="cpu", weights_only=False)
        # meta might be either the full metadata dict, or {'input_stats', 'output_stats'}
        if "input_stats" in meta:
            stats = {"input_stats": meta["input_stats"], "output_stats": meta["output_stats"]}
        else:
            stats = meta
    else:
        stats = compute_stats(data_list)

    data_norm    = apply_minmax(data_list, stats)
    data_mm      = compute_edge_attr_full(data_norm)              # for MGN/XMGN/GATv2/GraphUNet
    data_mm_ea   = add_augmented_edges_for_eagnn(data_mm, k_aug=16)  # for EA-GNN

    # Fixed 80/20/90-10 split
    labels = get_stratify_labels(data_list, args.stratify_key)
    tr_idx, va_idx, te_idx = stratified_train_val_test_split(
        len(data_mm), labels, test_frac=0.20, val_frac_of_train=0.10, seed=args.seed
    )
    print(f"split  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    test_data     = [data_mm[i]     for i in te_idx]
    test_data_ea  = [data_mm_ea[i]  for i in te_idx]

    test_loader     = DataLoader(test_data,    batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader_ea  = DataLoader(test_data_ea, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # -------- Load checkpoints ---------------------------------------------
    # Prefer new baseline_runs/<name>/model.pt (from train_baselines.py),
    # fall back to flat --ckpts_dir/<Class>_trained.pt (legacy notebook layout).
    def _resolve(name):
        legacy = {
            "mgn":    "MGN_trained.pt",
            "xmgn":   "XMGN_trained.pt",
            "eagnn":  "EAGNN_trained.pt",
            "gunet":  "GraphUNet_trained.pt",
        }
        # 1. new layout
        cand = os.path.join(args.ckpts_dir, name, "model.pt")
        if os.path.isfile(cand):
            return cand
        # 2. legacy layout
        if name in legacy:
            cand = os.path.join(args.ckpts_dir, legacy[name])
            if os.path.isfile(cand):
                return cand
        return None

    ckpt_map = {name: _resolve(name) for name in ("mgn", "xmgn", "eagnn", "gunet")}
    ckpt_map["gatv2"]  = args.gatv2_ckpt
    ckpt_map["dg3net"] = args.dg3_ckpt

    reports = {}
    per_sample_bundle = {}

    for name in args.models:
        path = ckpt_map.get(name)
        if path is None or not os.path.isfile(path):
            print(f"[{name}] checkpoint not found ({path}); skipping.")
            continue

        print(f"\n[{name}] loading checkpoint {path}")
        model = torch.load(path, map_location=device, weights_only=False).to(device)

        loader = test_loader_ea if name == "eagnn" else test_loader
        preds, trues = infer_and_collect(model, loader, device, stats, norm_mode="minmax")
        report = evaluate_over_set(preds, trues)

        reports[name] = {
            "aggregate":   report["aggregate"],
            "n_test":      len(preds),
            "checkpoint":  path,
        }
        per_sample_bundle[name] = {
            "per_sample": report["per_sample"],
            "preds":      [p.numpy() for p in preds],
            "trues":      [t.numpy() for t in trues],
            "test_idx":   te_idx,
        }
        print(format_aggregate_table(report, name=name))

    # -------- Save aggregate report ----------------------------------------
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump({
            "config":  vars(args),
            "reports": reports,
        }, f, indent=2)

    with open(os.path.join(args.out, "per_sample.pkl"), "wb") as f:
        pickle.dump(per_sample_bundle, f)

    # Pretty aggregate table to console
    print("\n===== AGGREGATE TABLE =====")
    for name, rep in reports.items():
        print(format_aggregate_table({"aggregate": rep["aggregate"]}, name=name))

    print(f"\nSaved reports to {args.out}")


if __name__ == "__main__":
    main()
