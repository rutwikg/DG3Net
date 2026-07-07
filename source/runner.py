"""
runner.py
=========
Shared training/eval driver for ablation variants V0..V4 and DG3Net.

Each ablation_vX.py sets a small config dict and calls run_ablation(...).
This keeps V0..V4 scripts short and prevents drift between them.

Now supports:
  * --output_mode {disp, sspe, all}  which channels of data.y to predict
  * --use_aug_edges                  add kNN augmented edges (EA-GNN-style,
                                     used by FEMAT block when use_femat_first)
  * --adaptive_lambda_E {on, off}    precompute lambda_E from data statistics
                                     using compute_lambda scheme (before ep 1)
  * --lambda_E_offset                offset for compute_lambda (default 1.0)
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
    evaluate_over_set,
    stratified_train_val_test_split,
    format_aggregate_table,
)
from checkpoint import save_checkpoint, load_checkpoint


# =============================================================================
# OUTPUT MODE  -> which channels of data.y to keep + which slices for aux losses
# =============================================================================

OUTPUT_MODES = {
    # mode name : (channel_indices_to_keep, disp_slice, sspe_slice)
    # slices are LOCAL to the KEPT tensor after slicing data.y
    "disp":  (list(range(0, 3)),   slice(0, 3),  None),
    "sspe":  (list(range(3, 21)),  None,          slice(0, 18)),
    "all":   (list(range(0, 21)),  slice(0, 3),   slice(3, 21)),
}


def build_argparser(variant):
    p = argparse.ArgumentParser(description=f"DG3Net ablation {variant}")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--root", type=str, required=True,
                   help="directory of .pt PyG Data files")
    p.add_argument("--out",  type=str, default="./runs")
    p.add_argument("--input_dim",  type=int, default=4)

    p.add_argument("--output_mode", type=str, default="disp",
                   choices=list(OUTPUT_MODES.keys()),
                   help="disp = u,v,w only; sspe = stress+strain+plastic strain only; "
                        "all = disp + sspe (21 channels).")

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--heads",      type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs",     type=int, default=500)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--stratify_key", type=str, default=None)

    p.add_argument("--lambda_E",   type=float, default=None,
                   help="Fixed gradient-loss weight (overrides variant default). "
                        "Ignored if --adaptive_lambda_E=on.")
    p.add_argument("--lambda_M",   type=float, default=None,
                   help="Constitutive-loss weight (overrides variant default).")

    p.add_argument("--adaptive_lambda_E", type=str, default="off",
                   choices=["on", "off"],
                   help="If on, lambda_E is computed once before training via "
                        "the compute_lambda scheme and held fixed. "
                        "Requires output_mode with a displacement slice.")
    p.add_argument("--lambda_E_offset", type=float, default=1.0,
                   help="Offset in compute_lambda: floor(log10 l_disp) - floor(log10 l_strain) + offset.")

    p.add_argument("--use_aug_edges", type=str, default="off",
                   choices=["on", "off"],
                   help="Add kNN augmented edges to each graph. Currently used "
                        "only by the FEMAT block (when use_femat_first=True).")
    p.add_argument("--k_aug",         type=int, default=16,
                   help="Number of augmented neighbours per node (kNN).")

    p.add_argument("--ckpt_every", type=int, default=50)
    p.add_argument("--resume",     type=str, default="auto",
                   choices=["auto", "yes", "no"])
    p.add_argument("--log_every",  type=int, default=1)
    return p


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


def prepare_data(args, use_geom_norm, output_channel_indices, use_aug_edges):
    """
    Load data, slice y to requested channels, normalize, and attach augmented
    edges if requested.
    """
    data_list = load_data_list(args.root)
    print(f"loaded {len(data_list)} graphs")

    ch = output_channel_indices
    for d in data_list:
        if d.y.ndim == 3:
            d.y = d.y[:, -1, :]                 # take last timestep, keep all channels
        # keep only requested channel indices
        d.y = d.y[:, ch]
        if d.x.shape[1] > args.input_dim:
            d.x = torch.cat([d.x[:, :3], d.x[:, -1:]], dim=1)

    stats = compute_stats(data_list)
    if use_geom_norm:
        data_norm = apply_geom_aware(data_list, stats)
        print("normalization: geometry-aware (per-graph coords + minmax rest)")
    else:
        data_norm = apply_minmax(data_list, stats)
        print("normalization: dataset-wide minmax")

    if use_aug_edges:
        data_norm = add_augmented_edges_to_dataset(data_norm, k_aug=args.k_aug)
        print(f"augmented edges attached (k={args.k_aug})")

    labels = get_stratify_labels(data_list, args.stratify_key)
    tr_idx, va_idx, te_idx = stratified_train_val_test_split(
        len(data_norm), labels, test_frac=0.20, val_frac_of_train=0.10, seed=args.seed
    )
    print(f"split  train={len(tr_idx)}  val={len(va_idx)}  test={len(te_idx)}")

    train_data = [data_norm[i] for i in tr_idx]
    val_data   = [data_norm[i] for i in va_idx]
    test_data  = [data_norm[i] for i in te_idx]
    return data_norm, stats, train_data, val_data, test_data, tr_idx, va_idx, te_idx


def run_ablation(variant, use_femat_first, use_geom_norm,
                 default_lambda_E, default_lambda_M,
                 default_use_aug_edges=False):
    parser = build_argparser(variant)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    channels, disp_slice, sspe_slice = OUTPUT_MODES[args.output_mode]
    output_dim = len(channels)

    lam_E = default_lambda_E if args.lambda_E is None else args.lambda_E
    lam_M = default_lambda_M if args.lambda_M is None else args.lambda_M

    if lam_M > 0 and sspe_slice is None:
        print(f"[{variant}] output_mode={args.output_mode} has no SSPE slice; disabling L_M.")
        lam_M = 0.0
    if lam_E > 0 and disp_slice is None:
        print(f"[{variant}] output_mode={args.output_mode} has no displacement slice; disabling L_E.")
        lam_E = 0.0

    use_aug_edges = (args.use_aug_edges == "on") or default_use_aug_edges
    if use_aug_edges and not use_femat_first:
        print(f"[{variant}] NOTE: --use_aug_edges=on but use_femat_first=False; "
              f"augmented edges attached to data but model doesn't consume them.")

    out_dir = os.path.join(args.out, variant)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{variant}] device={device}  out={out_dir}")
    print(f"[{variant}] output_mode={args.output_mode} (dim={output_dim})  "
          f"disp_slice={disp_slice}  sspe_slice={sspe_slice}")
    print(f"[{variant}] lam_E={lam_E}  lam_M={lam_M}  adaptive_E={args.adaptive_lambda_E}  "
          f"aug_edges={use_aug_edges}  femat={use_femat_first}  geom={use_geom_norm}")

    # -------------------- Data --------------------
    (data_norm, stats,
     train_data, val_data, test_data,
     tr_idx, va_idx, te_idx) = prepare_data(args, use_geom_norm, channels, use_aug_edges)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False, drop_last=False)

    # -------------------- Model --------------------
    model = GATv2Backbone(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim,
        output_dim=output_dim, heads=args.heads,
        use_femat_first=use_femat_first,
        use_aug_edges=use_aug_edges,
        pos_dim=3, mat_dim=max(1, args.input_dim - 3), dof_dim=6,
    ).to(device)
    print(f"[{variant}] params={count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # -------------------- Resume (if any) --------------------
    if args.resume == "no":
        rolling = os.path.join(out_dir, "checkpoint_last.pt")
        if os.path.isfile(rolling):
            print(f"[{variant}] --resume=no: removing existing {rolling}")
            os.remove(rolling)
        start_epoch, history, best_val, best_epoch = 1, {"train": [], "train_mse": [], "val": [], "l_E": [], "l_M": []}, float("inf"), -1
    else:
        start_epoch, history, best_val, best_epoch = load_checkpoint(
            out_dir, model, optimizer, scheduler, device
        )
        if args.resume == "yes" and start_epoch == 1:
            raise FileNotFoundError(f"--resume=yes but no checkpoint_last.pt in {out_dir}")
        history.setdefault("train_mse", [])
        if start_epoch > args.epochs:
            print(f"[{variant}] checkpoint at epoch {start_epoch-1} >= {args.epochs}; skipping training")

    # -------------------- Precompute lambda_E once before training --------------------
    lam_E_diag = None
    if args.adaptive_lambda_E == "on" and disp_slice is not None:
        if start_epoch == 1:
            print(f"[{variant}] adaptive_lambda_E=on -> precomputing lambda_E once before ep 1 "
                  f"(offset={args.lambda_E_offset})")
            lam_E, lam_E_diag = precompute_lambda_E(
                model, train_loader, criterion, device,
                disp_slice=disp_slice, offset=args.lambda_E_offset,
            )
            print(f"[{variant}] precomputed lam_E={lam_E:.3e}  "
                  f"(l_disp={lam_E_diag['l_disp_mean']:.2e}, "
                  f"l_strain={lam_E_diag['l_strain_mean']:.2e}, "
                  f"n_batches={lam_E_diag['n_batches_sampled']})")
        else:
            # Restore precomputed lam_E from checkpoint meta so a resumed run
            # continues with the SAME lam_E (not a fresh recomputation).
            try:
                state = torch.load(os.path.join(out_dir, "checkpoint_last.pt"),
                                    map_location="cpu", weights_only=False)
                saved_lam = state.get("meta", {}).get("lam_E")
                if saved_lam is not None:
                    lam_E = float(saved_lam)
                    print(f"[{variant}] resumed lam_E={lam_E:.3e} from checkpoint meta")
            except Exception as e:
                print(f"[{variant}] warning: could not restore lam_E from checkpoint ({e})")

    # -------------------- Train --------------------
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        t_ep = time.time()

        tr = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            lambda_E=lam_E, lambda_M=lam_M,
            epoch=epoch, total_epochs=args.epochs, show_progress=True,
            disp_slice=disp_slice, sspe_slice=sspe_slice,
        )
        _, _, val_mse = evaluate(model, val_loader, criterion, device, stats, norm_mode="minmax")
        scheduler.step()

        history["train"].append(tr["total"])
        history["train_mse"].append(tr["l_disp"])
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
            meta={"variant": variant, "use_femat_first": use_femat_first,
                  "use_geom_norm": use_geom_norm,
                  "use_aug_edges": use_aug_edges,
                  "lam_E": lam_E, "lam_M": lam_M,
                  "output_mode": args.output_mode},
        )

        ep_time = time.time() - t_ep
        eta_s   = ep_time * (args.epochs - epoch)
        if epoch == start_epoch or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
            print(f"[{variant}] ep{epoch:4d}/{args.epochs}  "
                  f"train_mse={tr['l_disp']:.4e}  val_mse={val_mse:.4e}  "
                  f"train_tot={tr['total']:.4e}  "
                  f"lE={tr['l_E']:.2e}  lM={tr['l_M']:.2e}  "
                  f"best_val={best_val:.4e}@{best_epoch}  "
                  f"ep_time={ep_time:.1f}s  ETA={eta_s/3600:.1f}h",
                  flush=True)

    wall = time.time() - t0

    # -------------------- Final eval on TEST --------------------
    model.load_state_dict(torch.load(os.path.join(out_dir, "model_best.pt"), map_location=device))
    preds_test, trues_test, test_mse_norm = evaluate(
        model, test_loader, criterion, device, stats, norm_mode="minmax"
    )
    report = evaluate_over_set(preds_test, trues_test)

    torch.save(model, os.path.join(out_dir, "model.pt"))
    torch.save(preds_test, os.path.join(out_dir, "predictions_test.pt"))
    torch.save(trues_test, os.path.join(out_dir, "ground_truth_test.pt"))
    torch.save({"train_idx": tr_idx, "val_idx": va_idx, "test_idx": te_idx},
               os.path.join(out_dir, "splits.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump({
            "variant":              variant,
            "use_femat_first":      use_femat_first,
            "use_geom_norm":        use_geom_norm,
            "use_aug_edges":        use_aug_edges,
            "output_mode":          args.output_mode,
            "channels_kept":        channels,
            "adaptive_lambda_E":    args.adaptive_lambda_E,
            "lambda_E":             lam_E,
            "lambda_M":             lam_M,
            "lambda_E_diag":        lam_E_diag,
            "config":               vars(args),
            "n_params":             count_params(model),
            "wall_hours":           wall / 3600,
            "best_val_epoch":       best_epoch,
            "best_val_mse":         best_val,
            "test_mse_normalized":  test_mse_norm,
            "aggregate":            report["aggregate"],
            "per_sample":           report["per_sample"],
        }, f, indent=2)

    print(f"[{variant}] wall={wall/3600:.2f}h  best val@{best_epoch}: {best_val:.4e}")
    print(format_aggregate_table(report, name=variant))
    print(f"[{variant}] outputs -> {out_dir}")
