"""
ablation_v4.py — V4: FULL DG3Net
    - 1x FEMAT + 4x GATv2 blocks
    - FEMAT runs on BOTH mesh edges AND kNN augmented edges (EA-GNN-style
      parallel path). Outputs summed.
    - geometry-aware normalization
    - MSE + adaptive L_E + constitutive L_M

If your .pt files lack data.dof or data.bc, FEMAT degrades gracefully:
    - dof=None -> DOF-compatibility term is zeroed out
    - bc=None  -> BC gate off (no residual gating)

Usage (recommended, all channels):
    python ablation_v4.py --gpu 0 --root /path/graphs --output_mode all

Displacement-only ablation (recommended first pass while iterating):
    python ablation_v4.py --gpu 0 --root /path/graphs --output_mode disp
    (L_M auto-disabled because sspe channels are missing)
"""
import sys
from runner import run_ablation

if "--adaptive_lambda_E" not in sys.argv:
    sys.argv.extend(["--adaptive_lambda_E", "on"])
# V4 uses augmented edges by default; user can force off with --use_aug_edges off
if "--use_aug_edges" not in sys.argv:
    sys.argv.extend(["--use_aug_edges", "on"])

if __name__ == "__main__":
    run_ablation(
        variant                = "v4",
        use_femat_first        = True,
        use_geom_norm          = True,
        default_lambda_E       = 0.0,
        default_lambda_M       = 1e-4,
        default_use_aug_edges  = True,
    )
