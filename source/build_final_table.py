"""
build_final_table.py
====================
Merges metrics from:
    - ./runs/v0..v4, v4a       (ablation variants)
    - ./baseline_runs/mgn..gunet   (literature baselines)

into one JSON + one paste-ready LaTeX table row set for the paper.

Reads the metrics_test.json each training script already saved. Does NOT
re-run inference. Assumes every subdirectory has:
    <dir>/metrics_test.json      containing aggregate + per_sample dicts

Usage:
    python build_final_table.py \
      --runs_dir     ./runs \
      --baselines_dir ./baseline_runs \
      --out          ./final_report

Writes:
    <out>/final_metrics.json          full aggregate + per-sample
    <out>/final_metrics.csv           tabular summary
    <out>/table_baselines.tex         LaTeX rows for the SOTA comparison
    <out>/table_ablation.tex          LaTeX rows for the ablation study
    <out>/per_sample.pkl              per-sample bundle for plots
"""
import argparse, os, json, pickle, glob
from collections import OrderedDict


ABLATION_ORDER = ["v0", "v1", "v2", "v3", "v4", "v4a"]
ABLATION_LABELS = {
    "v0":  "V0: GATv2 baseline",
    "v1":  "V1: + Geometry-aware norm",
    "v2":  r"V2: + Grad. consistency $\mathcal{L}_{E}$",
    "v3":  r"V3: + Constitutive reg. $\mathcal{L}_{M}$",
    "v4":  "V4: + FEMAT (full DG3Net)",
    "v4a": "V4a: FEMAT + norm (no aux losses)",
}
BASELINE_ORDER = ["mgn", "xmgn", "eagnn", "gunet"]
BASELINE_LABELS = {
    "mgn":   r"MeshGraphNet \cite{pfaff2020}",
    "xmgn":  r"X-MeshGraphNet \cite{xmeshgraphnet}",
    "eagnn": r"EA-GNN \cite{gladstone2024eagnn}",
    "gunet": r"Graph U-Net \cite{gaograph}",
}


def load_metrics(json_path):
    with open(json_path) as f:
        return json.load(f)


def fmt(x, prec=3):
    """Format for LaTeX cell: 3-sig-fig scientific if small, decimal otherwise."""
    if x is None: return "---"
    if abs(x) < 1e-2 or abs(x) > 1e3:
        return f"${x:.{prec}e}$".replace("e-0", "e-").replace("e+0", "e")
    return f"${x:.{prec}f}$"


def scan_dir(base_dir, name_order, labels):
    """Return dict: name -> {aggregate, per_sample, path, exists, n_params, wall_hours}."""
    out = OrderedDict()
    for name in name_order:
        sub = os.path.join(base_dir, name)
        mf  = os.path.join(sub, "metrics_test.json")
        if not os.path.isfile(mf):
            out[name] = {"exists": False, "path": mf, "label": labels[name]}
            continue
        m = load_metrics(mf)
        out[name] = {
            "exists":      True,
            "path":        mf,
            "label":       labels[name],
            "aggregate":   m.get("aggregate", {}),
            "per_sample":  m.get("per_sample", {}),
            "n_params":    m.get("n_params"),
            "wall_hours":  m.get("wall_hours"),
            "best_val_epoch": m.get("best_val_epoch"),
            "best_val_mse":   m.get("best_val_mse"),
        }
    return out


def write_latex_table_rows(entries, out_path, kind):
    """
    Write only the TABULAR ROWS (no \\begin{table} ... \\end{table}).
    Paste the rows into the table skeleton already in your tex file.
    """
    cols = ["test_mse_mean", "rel_l2_mean", "peak_err_mean",
            "max_nodal_error", "top5_rel_l2_mean"]
    with open(out_path, "w") as f:
        f.write(f"% Auto-generated rows for {kind} table\n")
        for name, e in entries.items():
            if not e["exists"]:
                f.write(f"% {name}: metrics_test.json NOT FOUND at {e['path']}\n")
                continue
            a = e["aggregate"]
            row = " & ".join([fmt(a.get(c)) for c in cols])
            f.write(f"{e['label']} & {row} \\\\ \\hline\n")


def write_csv(entries_ablation, entries_baseline, out_path):
    with open(out_path, "w") as f:
        header = ["variant", "type", "label",
                  "test_mse_mean", "test_mse_std",
                  "rel_l2_mean",   "rel_l2_std",
                  "peak_err_mean", "peak_err_std",
                  "max_nodal_error", "max_nodal_err_mean",
                  "top5_rel_l2_mean", "top5_rel_l2_std",
                  "r2_mean", "r2_std",
                  "n_params", "wall_hours",
                  "best_val_epoch", "best_val_mse"]
        f.write(",".join(header) + "\n")
        for kind, entries in [("ablation", entries_ablation), ("baseline", entries_baseline)]:
            for name, e in entries.items():
                if not e["exists"]:
                    f.write(f"{name},{kind},NOT_FOUND\n")
                    continue
                a = e["aggregate"]
                row = [name, kind, e["label"].replace(",", ";")]
                for c in header[3:15]:
                    row.append(str(a.get(c, "")))
                row.append(str(e.get("n_params", "")))
                row.append(str(e.get("wall_hours", "")))
                row.append(str(e.get("best_val_epoch", "")))
                row.append(str(e.get("best_val_mse", "")))
                f.write(",".join(row) + "\n")


def write_per_sample_bundle(entries_ablation, entries_baseline, out_path):
    bundle = {}
    for name, e in {**entries_ablation, **entries_baseline}.items():
        if not e["exists"]:
            continue
        bundle[name] = {"per_sample": e["per_sample"], "label": e["label"]}
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)


def print_summary(entries_ablation, entries_baseline):
    hdr = f"{'variant':<18s} {'exists':<7s} {'test_mse':<12s} {'rel_l2':<12s} {'peak_err':<12s} {'max_node':<12s} {'top5':<12s} {'params':<10s} {'wall_h':<7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for kind, entries in [("ablation", entries_ablation), ("baseline", entries_baseline)]:
        print(f"--- {kind} ---")
        for name, e in entries.items():
            if not e["exists"]:
                print(f"{name:<18s} NO   (missing {e['path']})")
                continue
            a = e["aggregate"]
            print(f"{name:<18s} yes  "
                  f"{a.get('test_mse_mean',0):<12.4e} "
                  f"{a.get('rel_l2_mean',0):<12.4e} "
                  f"{a.get('peak_err_mean',0):<12.4e} "
                  f"{a.get('max_nodal_error',0):<12.4e} "
                  f"{a.get('top5_rel_l2_mean',0):<12.4e} "
                  f"{e.get('n_params') or 0:<10d} "
                  f"{e.get('wall_hours') or 0:<7.2f}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir",      type=str, default="./runs")
    p.add_argument("--baselines_dir", type=str, default="./baseline_runs")
    p.add_argument("--out",           type=str, default="./final_report")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)

    entries_ablation = scan_dir(args.runs_dir,      ABLATION_ORDER, ABLATION_LABELS)
    entries_baseline = scan_dir(args.baselines_dir, BASELINE_ORDER, BASELINE_LABELS)

    # Aggregate JSON
    payload = {
        "ablation": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                     for k, v in entries_ablation.items()},
        "baseline": {k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
                     for k, v in entries_baseline.items()},
    }
    with open(os.path.join(args.out, "final_metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

    # CSV summary
    write_csv(entries_ablation, entries_baseline,
              os.path.join(args.out, "final_metrics.csv"))

    # LaTeX rows (paste into existing table skeletons in the .tex file)
    write_latex_table_rows(entries_ablation, os.path.join(args.out, "table_ablation.tex"), "ablation")
    write_latex_table_rows(entries_baseline, os.path.join(args.out, "table_baselines.tex"), "baseline")

    # Per-sample bundle for plotting
    write_per_sample_bundle(entries_ablation, entries_baseline,
                            os.path.join(args.out, "per_sample.pkl"))

    print_summary(entries_ablation, entries_baseline)
    print(f"\nFinal report saved to {args.out}")
    print(f"  final_metrics.json   full aggregate")
    print(f"  final_metrics.csv    spreadsheet-friendly")
    print(f"  table_ablation.tex   paste into ablation table in your .tex")
    print(f"  table_baselines.tex  paste into SOTA comparison table")
    print(f"  per_sample.pkl       feed to make_plots.py for scatter/hist/CDF")


if __name__ == "__main__":
    main()
