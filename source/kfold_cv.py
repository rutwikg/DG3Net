"""
kfold_cv.py
===========
K-fold cross-validation runner, updated for current common.py.

Supports variants V0..V4a, output modes {disp, sspe, all}, augmented edges
(for V4/V4a), and adaptive lambda_E (for V2/V3/V4).

Usage (all folds sequentially on one GPU):
    python kfold_cv.py --variant v4a --k 5 --gpu 0 \
                       --root /path/to/graphs --output_mode disp --epochs 500

Usage (parallel across GPUs, one fold per GPU):
    python kfold_cv.py --variant v4a --k 5 --fold_id 0 --gpu 0 --root ... &
    python kfold_cv.py --variant v4a --k 5 --fold_id 1 --gpu 1 --root ... &
    python kfold_cv.py --variant v4a --k 5 --fold_id 2 --gpu 2 --root ... &
    python kfold_cv.py --variant v4a --k 5 --fold_id 3 --gpu 3 --root ... &
    wait
    python kfold_cv.py --variant v4a --k 5 --fold_id 4 --gpu 0 --root ...

Notes:
    - When single-fold jobs are launched, cv_summary.json is NOT written
      automatically (individual jobs don't know when siblings are done).
      Run kfold_cv.py --aggregate_only ... after all folds are done to
      collect them.
    - Each fold seeds with (base_seed + fold_id) so folds are decorrelated.
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from common import (
    GATv2Backbone,
    apply_minmax, apply_geom_aware,
    load_data_list, compute_stats,
    train_one_epoch, evaluate,
    count_params,
    add_augmented_edges_to_dataset,
    precompute_lambda_E,
)
from metrics import (
    kfold_indices, evaluate_over_set, format_aggregate_table,
)
from checkpoint import save_checkpoint, load_checkpoint


# =============================================================================
# Same config table as runner.py so V4a etc. behave identically
# =============================================================================
VARIANT_CONFIG = {
    "v0":  dict(use_femat=False, use_geom=False, use_aug=False,
                lam_E=0.0, lam_M=0.0, adaptive_E=False),
    "v1":  dict(use_femat=False, use_geom=True,  use_aug=False,
                lam_E=0.0, lam_M=0.0, adaptive_E=False),
    "v2":  dict(use_femat=False, use_geom=True,  use_aug=False,
                lam_E=1e-3, lam_M=0.0, adaptive_E=True),
    "v3":  dict(use_femat=False, use_geom=True,  use_aug=False,
                lam_E=1e-3, lam_M=1e-4, adaptive_E=True),
    "v4":  dict(use_femat=True,  use_geom=True,  use_aug=True,
                lam_E=1e-3, lam_M=1e-4, adaptive_E=True),
    "v4a": dict(use_femat=True,  use_geom=True,  use_aug=True,
                lam_E=0.0, lam_M=0.0, adaptive_E=False),
}

OUTPUT_MODES = {
    "disp":  (list(range(0, 3)),   slice(0, 3),  None),
    "sspe":  (list(range(3, 21)),  None,          slice(0, 18)),
    "all":   (list(range(0, 21)),  slice(0, 3),   slice(3, 21)),
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, choices=list(VARIANT_CONFIG.keys()), required=True)
    p.add_argument("--gpu",     type=int, default=0)
    p.add_argument("--root",    type=str, required=True)
    p.add_argument("--out",     type=str, default="./cv_runs")
    p.add_argument("--k",       type=int, default=5)
    p.add_argument("--fold_id", type=int, default=None,
                   help="If set, run ONLY this fold (0..k-1). Useful for GPU parallel.")
    p.add_argument("--output_mode", type=str, default="disp",
                   choices=list(OUTPUT_MODES.keys()))
    p.add_argument("--input_dim",  type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--heads",      type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int, default=42,
                   help="Base seed used both for fold generation and per-fold model init "
                        "(per-fold init uses seed + fold_id).")
    p.add_argument("--stratify_key", type=str, default=None)
    p.add_argument("--k_aug",        type=int, default=16)
    p.add_argument("--ckpt_every",   type=int, default=50)
    p.add_argument("--resume",       type=str, default="auto",
                   choices=["auto", "yes", "no"])
    p.add_argument("--log_every",    type=int, default=1)
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip training; scan <out>/<variant>/cv<k>/fold_*/metrics_test.json "
                        "and write cv_summary.json.")
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


def prepare_data(args, cfg, channels):
    data_list = load_data_list(args.root)
    print(f"loaded {len(data_list)} graphs")

    for d in data_list:
        if d.y.ndim == 3:
            d.y = d.y[:, -1, :]
        d.y = d.y[:, channels]
        if d.x.shape[1] > args.input_dim:
            d.x = torch.cat([d.x[:, :3], d.x[:, -1:]], dim=1)

    stats = compute_stats(data_list)
    if cfg["use_geom"]:
        data_norm = apply_geom_aware(data_list, stats)
        print("normalization: geometry-aware")
    else:
        data_norm = apply_minmax(data_list, stats)
        print("normalization: dataset-wide minmax")

    if cfg["use_aug"]:
        data_norm = add_augmented_edges_for_v4(data_norm, args.k_aug)
        print(f"augmented edges attached (k={args.k_aug})")

    return data_list, data_norm, stats


def add_augmented_edges_for_v4(data_norm, k_aug):
    return add_augmented_edges_to_dataset(data_norm, k_aug=k_aug)


def train_fold(fold, tr_idx, te_idx, data_norm, stats, args, cfg, device, out_dir,
               disp_slice, sspe_slice, output_dim):
    print(f"\n===== FOLD {fold}  train={len(tr_idx)} test={len(te_idx)} =====")

    train_data = [data_norm[i] for i in tr_idx]
    test_data  = [data_norm[i] for i in te_idx]
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    fold_seed = args.seed + fold
    torch.manual_seed(fold_seed); np.random.seed(fold_seed)

    model = GATv2Backbone(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        output_dim=output_dim, heads=args.heads,
        use_femat_first=cfg["use_femat"],
        use_aug_edges=cfg["use_aug"],
        pos_dim=3, mat_dim=max(1, args.input_dim - 3), dof_dim=6,
    ).to(device)
    print(f"  fold{fold} params={count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    fold_dir = os.path.join(out_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # -------- Resume --------
    if args.resume == "no":
        rolling = os.path.join(fold_dir, "checkpoint_last.pt")
        if os.path.isfile(rolling):
            os.remove(rolling)
        start_epoch = 1
        history = {"train": [], "train_mse": [], "l_E": [], "l_M": []}
        best_val, best_epoch = float("inf"), -1
    else:
        start_epoch, history, best_val, best_epoch = load_checkpoint(
            fold_dir, model, optimizer, scheduler, device
        )
        history.setdefault("train_mse", [])
        if args.resume == "yes" and start_epoch == 1:
            raise FileNotFoundError(f"--resume=yes but no checkpoint in {fold_dir}")

    # -------- Resolve losses --------
    lam_E = cfg["lam_E"]
    lam_M = cfg["lam_M"]
    if lam_M > 0 and sspe_slice is None:
        print(f"  fold{fold}: output_mode has no SSPE slice; L_M disabled.")
        lam_M = 0.0
    if lam_E > 0 and disp_slice is None:
        print(f"  fold{fold}: output_mode has no displacement slice; L_E disabled.")
        lam_E = 0.0

    # Adaptive lambda_E (once at fold start)
    if cfg["adaptive_E"] and lam_E > 0 and start_epoch == 1:
        print(f"  fold{fold}: precomputing adaptive lambda_E...")
        lam_E, diag = precompute_lambda_E(
            model, train_loader, criterion, device,
            disp_slice=disp_slice, offset=1.0,
        )
        print(f"  fold{fold}: lam_E={lam_E:.3e} (l_disp={diag['l_disp_mean']:.2e}, "
              f"l_strain={diag['l_strain_mean']:.2e})")
    elif cfg["adaptive_E"] and lam_E > 0 and start_epoch > 1:
        try:
            state = torch.load(os.path.join(fold_dir, "checkpoint_last.pt"),
                                map_location="cpu", weights_only=False)
            saved = state.get("meta", {}).get("lam_E")
            if saved is not None:
                lam_E = float(saved)
                print(f"  fold{fold}: resumed lam_E={lam_E:.3e}")
        except Exception:
            pass

    # -------- Train --------
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()
        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            lambda_E=lam_E, lambda_M=lam_M,
            epoch=epoch, total_epochs=args.epochs, show_progress=True,
            disp_slice=disp_slice, sspe_slice=sspe_slice,
        )
        scheduler.step()

        history["train"].append(tr["total"])
        history["train_mse"].append(tr["l_disp"])
        history["l_E"].append(tr["l_E"])
        history["l_M"].append(tr["l_M"])

        # No held-out val in k-fold (test IS the held-out set).
        # Track best-train-MSE checkpoint as a proxy.
        if tr["l_disp"] < best_val:
            best_val, best_epoch = tr["l_disp"], epoch
            torch.save(model.state_dict(), os.path.join(fold_dir, "model_best.pt"))

        save_checkpoint(
            fold_dir, epoch, model, optimizer, scheduler,
            history, best_val, best_epoch,
            ckpt_every=args.ckpt_every,
            meta={"variant": args.variant, "fold": fold,
                  "lam_E": lam_E, "lam_M": lam_M},
        )

        ep_time = time.time() - t_ep
        eta_s = ep_time * (args.epochs - epoch)
        if epoch == start_epoch or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(f"  fold{fold} ep{epoch:4d}/{args.epochs}  train_mse={tr['l_disp']:.4e}  "
                  f"train_tot={tr['total']:.4e}  ep_time={ep_time:.1f}s  ETA={eta_s/3600:.1f}h",
                  flush=True)
    wall = time.time() - t0

    # -------- Final eval on held-out fold --------
    # Load best-train checkpoint (proxy for early-stopping since no val set)
    model.load_state_dict(torch.load(os.path.join(fold_dir, "model_best.pt"),
                                     map_location=device))
    preds, trues, mse_norm = evaluate(
        model, test_loader, criterion, device, stats, norm_mode="minmax"
    )
    report = evaluate_over_set(preds, trues)

    torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
    torch.save(preds, os.path.join(fold_dir, "predictions_test.pt"))
    torch.save(trues, os.path.join(fold_dir, "ground_truth_test.pt"))
    torch.save({"train_idx": tr_idx, "test_idx": te_idx},
               os.path.join(fold_dir, "splits.pt"))
    with open(os.path.join(fold_dir, "metrics_test.json"), "w") as f:
        json.dump({
            "variant":              args.variant,
            "fold":                 fold,
            "n_train":              len(tr_idx),
            "n_test":               len(te_idx),
            "output_mode":          args.output_mode,
            "config":               vars(args),
            "wall_hours":           wall / 3600,
            "best_train_epoch":     best_epoch,
            "best_train_mse":       best_val,
            "test_mse_normalized":  mse_norm,
            "aggregate":            report["aggregate"],
            "per_sample":           report["per_sample"],
        }, f, indent=2)

    print(f"  {format_aggregate_table(report, name=f'fold{fold}')}")
    return report["aggregate"]


def aggregate_folds(out_dir):
    """Scan fold_*/metrics_test.json and write cv_summary.json + summary.txt."""
    import glob
    fold_files = sorted(glob.glob(os.path.join(out_dir, "fold_*", "metrics_test.json")))
    if not fold_files:
        print(f"No fold files found in {out_dir}")
        return

    print(f"Found {len(fold_files)} folds")
    per_fold_aggs = []
    for f in fold_files:
        with open(f) as h:
            d = json.load(h)
        per_fold_aggs.append((d.get("fold"), d["aggregate"]))

    keys = list(per_fold_aggs[0][1].keys())
    summary = {}
    for k in keys:
        vals = np.array([agg[k] for _, agg in per_fold_aggs if agg.get(k) is not None])
        if len(vals) == 0:
            continue
        summary[k] = {
            "mean":     float(vals.mean()),
            "std":      float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
            "min":      float(vals.min()),
            "max":      float(vals.max()),
            "per_fold": vals.tolist(),
            "n_folds":  len(vals),
        }

    with open(os.path.join(out_dir, "cv_summary.json"), "w") as f:
        json.dump({"folds": [fi for fi, _ in per_fold_aggs], "summary": summary}, f, indent=2)

    print(f"\nCV summary  (n_folds={len(per_fold_aggs)})")
    print("-" * 80)
    for k in keys:
        if k not in summary:
            continue
        s = summary[k]
        print(f"  {k:25s}  {s['mean']:.4e} ± {s['std']:.2e}   "
              f"(range: {s['min']:.4e} .. {s['max']:.4e})")
    print(f"\nWritten to {out_dir}/cv_summary.json")


def main():
    args = parse()
    cfg = VARIANT_CONFIG[args.variant]
    channels, disp_slice, sspe_slice = OUTPUT_MODES[args.output_mode]
    output_dim = len(channels)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.out, args.variant, f"cv{args.k}")
    os.makedirs(out_dir, exist_ok=True)

    if args.aggregate_only:
        aggregate_folds(out_dir)
        return

    print(f"CV variant={args.variant}  output_mode={args.output_mode} "
          f"(dim={output_dim})  device={device}  k={args.k}  out={out_dir}")
    print(f"config: {cfg}")

    data_list, data_norm, stats = prepare_data(args, cfg, channels)

    labels = get_stratify_labels(data_list, args.stratify_key)
    splits = kfold_indices(len(data_norm), k=args.k, seed=args.seed,
                            stratify_labels=labels if args.stratify_key else None)

    fold_range = [args.fold_id] if args.fold_id is not None else range(args.k)

    per_fold_aggs = {}
    for fi in fold_range:
        sp = splits[fi]
        agg = train_fold(fi, sp["train_idx"], sp["test_idx"],
                          data_norm, stats, args, cfg, device, out_dir,
                          disp_slice, sspe_slice, output_dim)
        per_fold_aggs[fi] = agg

    # If we ran all folds, aggregate
    if args.fold_id is None:
        aggregate_folds(out_dir)


if __name__ == "__main__":
    main()
