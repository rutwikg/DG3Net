"""
train_baselines.py
==================
Retrain MeshGraphNet, X-MeshGraphNet, EA-GNN, and GraphUNet on the same
162 / 18 / 45 stratified split, same optimizer (AdamW, lr=1e-3, cosine),
same batch size (2), same epoch budget (500) as the ablation runs.

Each baseline gets its own subdirectory in --out; layout mirrors the
ablation scripts so downstream analysis code (metrics, plots) works
without modification.

Launch (parallel across 4 GPUs):

    python train_baselines.py --gpu 0 --which mgn    --root /path/graphs --output_dim 21 &
    python train_baselines.py --gpu 1 --which xmgn   --root /path/graphs --output_dim 21 &
    python train_baselines.py --gpu 2 --which eagnn  --root /path/graphs --output_dim 21 &
    python train_baselines.py --gpu 3 --which gunet  --root /path/graphs --output_dim 21 &
    wait

Or all four sequentially with --which all.

Each run writes to <out>/<name>/:
    model.pt              full model object
    model_best.pt         best-val state_dict
    history.json          training curves
    metrics_test.json     aggregate + per-sample metrics
    predictions_test.pt   physical predictions
    ground_truth_test.pt  physical ground truths
    splits.pt             train/val/test indices actually used
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common import (
    load_data_list, compute_stats, apply_minmax,
    train_one_epoch, evaluate, count_params,
)
from metrics import (
    evaluate_over_set, stratified_train_val_test_split, format_aggregate_table,
)
from baseline_models import (
    MeshGraphNet, XMeshGraphNet, EAGNN, GraphUNet,
    compute_edge_attr_full, add_augmented_edges,
)
from checkpoint import save_checkpoint, load_checkpoint


BASELINE_REGISTRY = {
    "mgn":     "MeshGraphNet",
    "xmgn":    "XMeshGraphNet",
    "eagnn":   "EAGNN",
    "gunet":   "GraphUNet",
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",         type=int, default=0)
    p.add_argument("--which",       type=str, choices=list(BASELINE_REGISTRY.keys()) + ["all"],
                   default="all")
    p.add_argument("--root",        type=str, required=True,
                   help="directory of .pt PyG Data files")
    p.add_argument("--out",         type=str, default="./baseline_runs")
    p.add_argument("--input_dim",   type=int, default=4)
    p.add_argument("--output_dim",  type=int, default=3)
    p.add_argument("--hidden_dim",  type=int, default=64)
    p.add_argument("--heads",       type=int, default=4)
    p.add_argument("--batch_size",  type=int, default=2)
    p.add_argument("--epochs",      type=int, default=500)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--stratify_key", type=str, default=None)
    p.add_argument("--mgn_layers",  type=int, default=15)
    p.add_argument("--xmgn_layers", type=int, default=15)
    p.add_argument("--eagnn_layers",type=int, default=10)
    p.add_argument("--gunet_depth", type=int, default=3)
    p.add_argument("--k_aug",       type=int, default=16)
    p.add_argument("--ckpt_every",  type=int, default=50)
    p.add_argument("--resume",      type=str, default="auto",
                   choices=["auto", "yes", "no"])
    p.add_argument("--log_every",   type=int, default=1)
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


def prepare_shared(args):
    """
    Load data, slim x/y, normalize, and produce a base data_mm list with
    the standard 8-dim edge_attr. EA-GNN additionally needs augmented edges.
    """
    data_list = load_data_list(args.root)
    for d in data_list:
        if d.y.ndim == 3:
            d.y = d.y[:, -1, :args.output_dim]
        else:
            d.y = d.y[:, :args.output_dim]
        if d.x.shape[1] > args.input_dim:
            d.x = torch.cat([d.x[:, :3], d.x[:, -1:]], dim=1)

    stats = compute_stats(data_list)
    data_norm = apply_minmax(data_list, stats)
    data_mm   = compute_edge_attr_full(data_norm)     # 8-dim edges

    labels = get_stratify_labels(data_list, args.stratify_key)
    tr_idx, va_idx, te_idx = stratified_train_val_test_split(
        len(data_mm), labels, test_frac=0.20, val_frac_of_train=0.10, seed=args.seed
    )
    return data_mm, stats, tr_idx, va_idx, te_idx


def build_model(name, args, device):
    hd = args.hidden_dim
    od = args.output_dim
    if name == "mgn":
        return MeshGraphNet(input_dim=args.input_dim, edge_dim=8,
                            hidden_dim=hd, output_dim=od,
                            num_mp_steps=args.mgn_layers).to(device)
    if name == "xmgn":
        return XMeshGraphNet(input_dim=args.input_dim, edge_dim=8,
                             hidden_dim=hd, output_dim=od,
                             num_mp_steps=args.xmgn_layers).to(device)
    if name == "eagnn":
        return EAGNN(input_dim=args.input_dim, edge_dim=8, aug_edge_dim=4,
                     hidden_dim=hd, output_dim=od,
                     num_mp_steps=args.eagnn_layers).to(device)
    if name == "gunet":
        return GraphUNet(input_dim=args.input_dim, hidden_dim=hd, output_dim=od,
                         depth=args.gunet_depth, pool_ratio=0.5).to(device)
    raise ValueError(name)


def train_one_baseline(name, args, device, data_mm, stats, tr_idx, va_idx, te_idx):
    print(f"\n===== TRAIN {name.upper()} =====")
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    # EA-GNN needs augmented edges; others use the standard 8-dim edges.
    if name == "eagnn":
        data_use = add_augmented_edges(data_mm, k_aug=args.k_aug)
    else:
        data_use = data_mm

    train_data = [data_use[i] for i in tr_idx]
    val_data   = [data_use[i] for i in va_idx]
    test_data  = [data_use[i] for i in te_idx]

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = build_model(name, args, device)
    print(f"[{name}] params={count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # -------- Resume --------
    if args.resume == "no":
        rolling = os.path.join(out_dir, "checkpoint_last.pt")
        if os.path.isfile(rolling):
            print(f"[{name}] --resume=no: removing existing {rolling}")
            os.remove(rolling)
        start_epoch, history, best_val, best_epoch = 1, {"train": [], "val": [], "l_E": [], "l_M": []}, float("inf"), -1
    else:
        start_epoch, history, best_val, best_epoch = load_checkpoint(
            out_dir, model, optimizer, scheduler, device
        )
        if args.resume == "yes" and start_epoch == 1:
            raise FileNotFoundError(f"--resume=yes but no checkpoint in {out_dir}")

    # -------- Train --------
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            lambda_E=0.0, lambda_M=0.0,  # baselines get NO physics losses
            epoch=epoch, total_epochs=args.epochs, show_progress=True,
        )
        _, _, val_mse = evaluate(model, val_loader, criterion, device, stats, norm_mode="minmax")
        scheduler.step()

        history["train"].append(tr["total"])
        history["val"].append(val_mse)
        history["l_E"].append(tr["l_E"])
        history["l_M"].append(tr["l_M"])

        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))

        save_checkpoint(
            out_dir, epoch, model, optimizer, scheduler,
            history, best_val, best_epoch,
            ckpt_every=args.ckpt_every,
            meta={"baseline": name, "class": BASELINE_REGISTRY[name]},
        )

        ep_time = time.time() - t_ep
        eta_s   = ep_time * (args.epochs - epoch)
        if epoch == start_epoch or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(f"[{name}] ep{epoch:4d}/{args.epochs}  "
                  f"train={tr['total']:.4e}  val={val_mse:.4e}  "
                  f"best={best_val:.4e}@{best_epoch}  "
                  f"ep_time={ep_time:.1f}s  ETA={eta_s/3600:.1f}h",
                  flush=True)

    wall = time.time() - t0

    # -------- Final eval on TEST --------------------------------------------
    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"), map_location=device))
    preds, trues, mse_norm = evaluate(model, test_loader, criterion, device, stats, norm_mode="minmax")
    report = evaluate_over_set(preds, trues)

    torch.save(model, os.path.join(out_dir, "model.pt"))
    torch.save(preds, os.path.join(out_dir, "predictions_test.pt"))
    torch.save(trues, os.path.join(out_dir, "ground_truth_test.pt"))
    torch.save({"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx},
               os.path.join(out_dir, "splits.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump({
            "baseline":            name,
            "class":               BASELINE_REGISTRY[name],
            "config":              vars(args),
            "n_params":            count_params(model),
            "wall_hours":          wall / 3600,
            "best_val_epoch":      best_epoch,
            "best_val_mse":        best_val,
            "test_mse_normalized": mse_norm,
            "aggregate":           report["aggregate"],
            "per_sample":          report["per_sample"],
        }, f, indent=2)

    print(format_aggregate_table(report, name=name))
    print(f"[{name}] wall={wall/3600:.2f}h  outputs -> {out_dir}")
    return report


def main():
    args = parse()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)
    print(f"device={device}  which={args.which}  out={args.out}")

    data_mm, stats, tr_idx, va_idx, te_idx = prepare_shared(args)
    print(f"loaded {len(data_mm)} graphs  split train={len(tr_idx)}/val={len(va_idx)}/test={len(te_idx)}")

    if args.which == "all":
        for name in BASELINE_REGISTRY.keys():
            train_one_baseline(name, args, device, data_mm, stats, tr_idx, va_idx, te_idx)
    else:
        train_one_baseline(args.which, args, device, data_mm, stats, tr_idx, va_idx, te_idx)


if __name__ == "__main__":
    main()
