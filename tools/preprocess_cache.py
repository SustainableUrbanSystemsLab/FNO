"""
Precompute the Transformer->FNO remap once and cache it as fp16.

Why
---
NpyDataset._remap_to_fno runs on every __getitem__ over an mmap'd 65 GB
X.npy. The remap is deterministic and depends only on that one sample, so it
can be done once up front. Writing the result as fp16 also halves the bytes
read per epoch off the (network) filesystem, which is the dominant per-epoch
cost. All four training pipelines auto-detect the cache (see
NpyDataset.__init__): if <data>/X_fno_fp16.npy and <data>/Y_fno_fp16.npy
exist and match the source sample count, they are used and the on-the-fly
remap is skipped.

Usage
-----
    python tools/preprocess_cache.py                # uses config.toml data path
    python tools/preprocess_cache.py --data data    # explicit source dir
    python tools/preprocess_cache.py --force        # rebuild even if cache exists

Run once on a login/compute node before submitting training jobs. Output is
~49 GB (X 32.5 -> 16.3 GB fp16, Y 16.3 -> 8.1 GB fp16 ... roughly half the
source). Safe to delete anytime; training falls back to on-the-fly remap.
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.utils.config_loader import load_config
from pipelines.train.distributed import NpyDataset


def resolve_data_dir(explicit):
    if explicit:
        return explicit
    cfg = load_config()
    paths = cfg.get("paths", {})
    for key in ("data_folder_ice", "data_folder_linux"):
        p = paths.get(key)
        if p:
            p = os.path.expanduser(p)
            if os.path.exists(os.path.join(p, "X.npy")):
                return p
    return "data"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="Directory holding X.npy / Y.npy (default: from config.toml)")
    ap.add_argument("--force", action="store_true", help="Rebuild even if cache files already exist")
    ap.add_argument("--chunk", type=int, default=200, help="Progress-report interval in samples")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    x_path = os.path.join(data_dir, "X.npy")
    y_path = os.path.join(data_dir, "Y.npy")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        sys.exit(f"[preprocess_cache] X.npy / Y.npy not found in {data_dir!r}")

    x_out = os.path.join(data_dir, "X_fno_fp16.npy")
    y_out = os.path.join(data_dir, "Y_fno_fp16.npy")

    # NpyDataset does the format detection + owns _remap_to_fno. Construct it
    # with augment=False; if it decides the data is already FNO-format (or a
    # cache is already present) there is nothing to do.
    ds = NpyDataset(x_path, y_path, augment=False)
    if not ds.needs_remap:
        print("[preprocess_cache] Source is already FNO-format (or cache already active). Nothing to do.")
        return

    if not args.force and os.path.exists(x_out) and os.path.exists(y_out):
        cx, cy = np.load(x_out, mmap_mode="r"), np.load(y_out, mmap_mode="r")
        if cx.shape[0] == ds.X.shape[0] and cy.shape[0] == ds.Y.shape[0]:
            print(f"[preprocess_cache] Cache already present and matches "
                  f"({cx.shape} / {cy.shape}). Use --force to rebuild.")
            return
        print("[preprocess_cache] Existing cache shape mismatch -- rebuilding.")

    N = ds.X.shape[0]
    xshape = (N,) + ds.X.shape[1:]
    yshape = (N,) + ds.Y.shape[1:]
    print(f"[preprocess_cache] Source {data_dir}: X{ds.X.shape} Y{ds.Y.shape}")
    print(f"[preprocess_cache] Writing fp16 cache:\n    {x_out}  {xshape}\n    {y_out}  {yshape}")

    # Write to .tmp then rename, so a killed job never leaves a half-written
    # cache that training would pick up.
    x_tmp, y_tmp = x_out + ".tmp", y_out + ".tmp"
    xo = np.lib.format.open_memmap(x_tmp, mode="w+", dtype=np.float16, shape=xshape)
    yo = np.lib.format.open_memmap(y_tmp, mode="w+", dtype=np.float16, shape=yshape)

    t0 = time.time()
    for i in range(N):
        rx = np.asarray(ds.X[i], dtype=np.float32)
        ry = np.asarray(ds.Y[i], dtype=np.float32)
        ox, oy = ds._remap_to_fno(rx, ry)
        xo[i] = ox.astype(np.float16)
        yo[i] = oy.astype(np.float16)
        if i % args.chunk == 0 or i == N - 1:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            eta = (N - i - 1) / max(rate, 1e-6)
            print(f"  {i + 1}/{N}  ({rate:.1f} samp/s, ETA {eta/60:.1f} min)", flush=True)

    xo.flush(); yo.flush()
    del xo, yo
    os.replace(x_tmp, x_out)
    os.replace(y_tmp, y_out)
    print(f"[preprocess_cache] Done in {(time.time() - t0)/60:.1f} min. "
          f"Training will now auto-detect the cache.")


if __name__ == "__main__":
    main()
