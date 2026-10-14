"""Shared helpers for the follow-up experiments (exp_noise.py, exp_orient.py, exp_rank.py)."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
from multiprocessing import Pool

PROCS = int(os.environ.get("PROCS", "4"))


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def run_jobs(jobs, work, out, key, procs=PROCS):
    """resumable: jobs whose key is already in `out` are skipped; each result is one JSON line."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {key(r) for r in load_jsonl(out)}
    todo = [j for j in jobs if key(j) not in done]
    print(f"{out}: {len(todo)} jobs to run ({len(done)} already done)", flush=True)
    with Pool(procs) as pool, open(out, "a") as f:
        for rec in pool.imap_unordered(work, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print({k: v for k, v in rec.items() if not isinstance(v, list)}, flush=True)


def boot_ci(d, seed=0, n=4000):
    """mean and 95% bootstrap interval of the mean of d."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    boots = [rng.choice(d, len(d)).mean() for _ in range(n)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)
