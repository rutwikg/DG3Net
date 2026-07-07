"""
checkpoint.py
=============
Save and resume full training state so a killed job continues where it left off.

A checkpoint at <out_dir>/checkpoint_last.pt contains:
    - epoch                : int, the LAST completed epoch (resume starts at epoch+1)
    - model_state          : model.state_dict()
    - optimizer_state      : optimizer.state_dict()
    - scheduler_state      : scheduler.state_dict()
    - history              : dict of lists (train/val/l_E/l_M)
    - best_val             : float
    - best_epoch           : int
    - rng                  : {torch, cuda, numpy} RNG states for exact reproducibility
    - meta                 : arbitrary dict (variant name, config, git hash, ...)

Two files are maintained side-by-side:
    checkpoint_last.pt     - overwritten every epoch (rolling)
    checkpoint_epoch_N.pt  - overwritten every --ckpt_every epochs (persistent snapshots)

Use save_checkpoint(...) at the end of each epoch. Use load_checkpoint(...) at
the top of the training loop; if a checkpoint exists it restores everything and
returns the epoch to resume FROM.

Atomic write: writes to <path>.tmp then renames, so a SIGKILL mid-write leaves
the previous checkpoint intact rather than a half-written one.
"""
import os
import numpy as np
import torch


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out_dir, epoch, model, optimizer, scheduler,
                    history, best_val, best_epoch,
                    ckpt_every=50, meta=None):
    """
    Save training state. Called at the end of every epoch.

    Overwrites <out_dir>/checkpoint_last.pt every epoch, and additionally
    writes <out_dir>/checkpoint_epoch_{epoch}.pt every `ckpt_every` epochs.
    """
    os.makedirs(out_dir, exist_ok=True)
    state = {
        "epoch":            int(epoch),
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict() if scheduler is not None else None,
        "history":          history,
        "best_val":         float(best_val),
        "best_epoch":       int(best_epoch),
        "rng": {
            "torch":  torch.get_rng_state(),
            "cuda":   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy":  np.random.get_state(),
        },
        "meta":             dict(meta or {}),
    }
    _atomic_save(state, os.path.join(out_dir, "checkpoint_last.pt"))
    if ckpt_every and epoch % ckpt_every == 0:
        _atomic_save(state, os.path.join(out_dir, f"checkpoint_epoch_{epoch}.pt"))


def load_checkpoint(out_dir, model, optimizer, scheduler, device):
    """
    If <out_dir>/checkpoint_last.pt exists, restore model + optimizer +
    scheduler + RNG state and return (start_epoch, history, best_val, best_epoch).

    Otherwise return (1, {'train':[], 'val':[], 'l_E':[], 'l_M':[]}, inf, -1).
    """
    default = (1, {"train": [], "val": [], "l_E": [], "l_M": []}, float("inf"), -1)
    path = os.path.join(out_dir, "checkpoint_last.pt")
    if not os.path.isfile(path):
        return default

    print(f"[resume] found {path}, restoring...")
    state = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    rng = state.get("rng", {})
    if "torch" in rng and rng["torch"] is not None:
        torch.set_rng_state(rng["torch"].cpu() if isinstance(rng["torch"], torch.Tensor) else rng["torch"])
    if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as e:
            print(f"[resume] warning: cuda RNG restore failed ({e}); continuing")
    if "numpy" in rng and rng["numpy"] is not None:
        try:
            np.random.set_state(rng["numpy"])
        except Exception as e:
            print(f"[resume] warning: numpy RNG restore failed ({e}); continuing")

    start_epoch = int(state["epoch"]) + 1
    history     = state["history"]
    best_val    = float(state["best_val"])
    best_epoch  = int(state["best_epoch"])
    print(f"[resume] resuming from epoch {start_epoch}  (best val so far {best_val:.4e} @ ep{best_epoch})")
    return start_epoch, history, best_val, best_epoch
