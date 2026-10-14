"""Generate the unlabeled pretraining corpus and the labeled downstream sets."""
import os
import time
import numpy as np
from multiprocessing import Pool
import world as W


def _pre(args):
    n, seed = args
    return W.build_pretrain(n, n_dyn=8, seed=seed)


def _down(args):
    n, seed = args
    return W.build_downstream(n, seed)


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs("data", exist_ok=True)
    with Pool(2) as pool:
        parts = pool.map(_pre, [(100, 1000 + i) for i in range(21)])   # 2100 shapes: 2000 train + 100 val
        off, merged = 0, {k: [] for k in ("shapes", "X", "v", "vd", "sid")}
        for p in parts:
            for k in ("shapes", "X", "v", "vd"):
                merged[k].append(p[k])
            merged["sid"].append(p["sid"] + off)
            off += len(p["shapes"])
        pre = {k: np.concatenate(v) for k, v in merged.items()}
        np.savez("data/pretrain.npz", **pre)
        print(f"pretrain: {pre['shapes'].shape[0]} shapes, {pre['X'].shape[0]} samples, "
              f"{pre['X'].nbytes / 1e6:.0f}MB positions  ({time.time() - t0:.0f}s)")
        parts = pool.map(_down, [(89, 5000 + i) for i in range(4)])        # 356 labeled shapes
        down = {k: np.concatenate([p[k] for p in parts]) if k != "grid" else parts[0][k] for k in parts[0]}
        np.savez("data/downstream.npz", **down)
        print(f"downstream: {down['shapes'].shape[0]} shapes  ({time.time() - t0:.0f}s)")
