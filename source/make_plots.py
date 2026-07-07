"""
make_plots.py
=============
Consumes the pickled per-sample data from either the baseline metrics run OR
an ablation run's metrics_test.json and produces:
    - true-vs-predicted scatter (peak displacement magnitude per sample)
    - relative-L2 error histograms per model
    - side-by-side error CDFs

Reviewer requests addressed:
    R8: "use true vs predicted scatter plots and complement them with error
         distribution plots"
    R3.3.2 partial: error stats disaggregated

Usage:
    python make_plots.py --input ./baseline_report/per_sample.pkl \
                          --outdir ./baseline_report/plots

    or

    python make_plots.py --input ./runs/v4/metrics_test.json \
                          --outdir ./runs/v4/plots \
                          --single_variant v4
"""
import argparse, json, os, pickle
import numpy as np
import matplotlib.pyplot as plt


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--single_variant", type=str, default=None,
                   help="If set, treat --input as a single-model metrics_test.json.")
    return p.parse_args()


def load_bundle(path, single_variant=None):
    if single_variant is not None:
        with open(path) as f:
            data = json.load(f)
        ps = data["per_sample"]
        return {single_variant: {"per_sample": ps}}
    with open(path, "rb") as f:
        return pickle.load(f)


def scatter_peak(bundle, outdir):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = np.inf, -np.inf
    for name, blob in bundle.items():
        ps = blob["per_sample"]
        t = np.asarray(ps["peak_true"])
        p = np.asarray(ps["peak_pred"])
        ax.scatter(t, p, s=28, alpha=0.75, label=name)
        lo = min(lo, t.min(), p.min())
        hi = max(hi, t.max(), p.max())
    lo, hi = lo * 0.95, hi * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y = x")
    ax.set_xlabel("True peak displacement magnitude [mm]")
    ax.set_ylabel("Predicted peak displacement magnitude [mm]")
    ax.set_title("Peak displacement: true vs predicted")
    ax.legend(loc="best", fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_peak_disp.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, "scatter_peak_disp.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def error_histograms(bundle, outdir, key="rel_l2", nbins=25):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    all_vals = []
    for name, blob in bundle.items():
        all_vals.extend(blob["per_sample"][key])
    lo, hi = min(all_vals), max(all_vals)
    bins = np.linspace(lo, hi, nbins + 1)
    for name, blob in bundle.items():
        v = np.asarray(blob["per_sample"][key])
        ax.hist(v, bins=bins, alpha=0.55, label=f"{name} (mean={v.mean():.3g})")
    ax.set_xlabel(f"Per-sample {key}")
    ax.set_ylabel("Number of test samples")
    ax.set_title(f"Distribution of {key} across the test set")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"hist_{key}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"hist_{key}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def error_cdf(bundle, outdir, key="rel_l2"):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, blob in bundle.items():
        v = np.sort(np.asarray(blob["per_sample"][key]))
        cdf = np.arange(1, len(v) + 1) / len(v)
        ax.plot(v, cdf, linewidth=2, label=name)
    ax.set_xlabel(f"Per-sample {key}")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(f"CDF of {key} across the test set")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"cdf_{key}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(outdir, f"cdf_{key}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse()
    os.makedirs(args.outdir, exist_ok=True)
    bundle = load_bundle(args.input, args.single_variant)

    scatter_peak(bundle, args.outdir)
    for key in ("rel_l2", "peak_err", "max_err", "top5"):
        error_histograms(bundle, args.outdir, key=key)
        error_cdf(bundle, args.outdir, key=key)
    print(f"Plots saved to {args.outdir}")


if __name__ == "__main__":
    main()
