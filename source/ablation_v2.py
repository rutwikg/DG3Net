"""
ablation_v2.py — V2: V1 + displacement-gradient consistency loss (adaptive)
    - 5x GATv2 blocks
    - geometry-aware normalization
    - MSE + Sobolev-style L_E with lambda_E precomputed via compute_lambda scheme
    - lambda_E is fixed after that (does NOT recompute each epoch)

To reproduce the notebook's compute_lambda behaviour, pass:
    --adaptive_lambda_E on --lambda_E_offset 1.0

Adaptive-λ is enabled by default. To disable, pass --adaptive_lambda_E off
and provide --lambda_E <value>.

Usage:
    python ablation_v2.py --gpu 2 --root /path/graphs --output_mode disp
"""
import sys
from runner import run_ablation

# Force adaptive_lambda_E=on by default. User can still override on CLI.
if "--adaptive_lambda_E" not in sys.argv:
    sys.argv.extend(["--adaptive_lambda_E", "on"])

if __name__ == "__main__":
    run_ablation(
        variant                = "v2",
        use_femat_first        = False,
        use_geom_norm          = True,
        default_lambda_E       = 0.0,   # placeholder; adaptive precompute overrides
        default_lambda_M       = 0.0,
        default_use_aug_edges  = False,
    )
