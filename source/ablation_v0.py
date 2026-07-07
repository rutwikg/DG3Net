"""
ablation_v0.py — V0: GATv2 baseline
    - 5x GATv2 blocks (no FEMAT)
    - dataset-wide min-max normalization
    - supervised MSE loss ONLY  (no gradient loss, no constitutive loss)
    - no augmented edges

Usage:
    python ablation_v0.py --gpu 0 --root /path/graphs --output_mode disp \
                          --epochs 500 --batch_size 2

Output mode options:
    --output_mode disp    predict u,v,w only  (default; matches paper primary claim)
    --output_mode sspe    predict stress+strain+plastic strain only (18 channels)
    --output_mode all     predict everything (21 channels)
"""
from runner import run_ablation

if __name__ == "__main__":
    run_ablation(
        variant                = "v0",
        use_femat_first        = False,
        use_geom_norm          = False,
        default_lambda_E       = 0.0,
        default_lambda_M       = 0.0,
        default_use_aug_edges  = False,
    )
