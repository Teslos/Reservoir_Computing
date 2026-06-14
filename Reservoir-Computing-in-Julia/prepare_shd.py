"""Pre-bin the Spiking Heidelberg Digits (SHD) into dense arrays for Julia.

SHD stores spikes as variable-length arrays of Float16 (units = 700 cochlear
channels, times in seconds), which HDF5.jl cannot read directly. This script
reads the raw SHD HDF5 with h5py, bins each sample's spikes into a dense
(channels x time) spike-count matrix (with optional channel pooling), optionally
subsamples per class, and writes a plain HDF5 (regular dense datasets) that
HDF5.jl reads without issue.

Output datasets in shd_binned.h5:
  X_train (N_tr, C, T), y_train (N_tr,), X_test (N_te, C, T), y_test (N_te,)

Usage:
  python Reservoir-Computing-in-Julia/prepare_shd.py [--pool 10] [--tbins 100]
         [--per-class 150] [--dur 1.0]
"""
import argparse, os
import numpy as np
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "shd")
N_CHAN = 700


def bin_split(path, pool, tbins, per_class, dur, rng):
    with h5py.File(path, "r") as f:
        times = f["spikes"]["times"]
        units = f["spikes"]["units"]
        labels = np.asarray(f["labels"], dtype=np.int64)

        # choose sample indices (optionally balanced per class)
        idx = np.arange(len(labels))
        if per_class > 0:
            sel = []
            for c in np.unique(labels):
                ci = idx[labels == c]
                rng.shuffle(ci)
                sel.append(ci[:per_class])
            idx = np.concatenate(sel)
            rng.shuffle(idx)

        C = N_CHAN // pool
        X = np.zeros((len(idx), C, tbins), dtype=np.float32)
        y = labels[idx].astype(np.int64)
        for k, i in enumerate(idx):
            t = np.asarray(times[i], dtype=np.float32)
            u = np.asarray(units[i], dtype=np.int64)
            tb = np.clip((t / dur * tbins).astype(np.int64), 0, tbins - 1)
            cb = np.clip(u // pool, 0, C - 1)
            np.add.at(X[k], (cb, tb), 1.0)
        return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=10)      # channels per input node
    ap.add_argument("--tbins", type=int, default=100)    # time bins over `dur`
    ap.add_argument("--per-class", type=int, default=150)  # samples/class (0 = all)
    ap.add_argument("--dur", type=float, default=1.0)    # seconds to bin over
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    Xtr, ytr = bin_split(os.path.join(DATA, "shd_train.h5"),
                         args.pool, args.tbins, args.per_class, args.dur, rng)
    Xte, yte = bin_split(os.path.join(DATA, "shd_test.h5"),
                         args.pool, args.tbins, max(1, args.per_class // 3), args.dur, rng)
    out = os.path.join(DATA, "shd_binned.h5")
    with h5py.File(out, "w") as f:
        f.create_dataset("X_train", data=Xtr, compression="gzip")
        f.create_dataset("y_train", data=ytr)
        f.create_dataset("X_test", data=Xte, compression="gzip")
        f.create_dataset("y_test", data=yte)
        f.attrs["pool"] = args.pool
        f.attrs["tbins"] = args.tbins
        f.attrs["channels"] = N_CHAN // args.pool
    print(f"wrote {out}")
    print(f"  X_train {Xtr.shape}, X_test {Xte.shape}, classes {len(np.unique(ytr))}")


if __name__ == "__main__":
    main()
