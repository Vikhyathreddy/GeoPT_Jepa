"""Generic resumable job runner (the run_all.py pattern).

    run(jobs, work, out, key=("variant", "task", "n", "seed"))

jobs is a list of dicts; work(job) -> dict runs one job in a worker process (it must be a
module-level function so it can be pickled).  Each result is stored as one JSON line
{**result, **job}, so the job's own fields always identify the row.  Jobs whose key already
appears in `out` are skipped, so an interrupted grid resumes where it stopped.  Every row must
carry per-body values (a list-valued field whose name starts with "per_"), for paired statistics
(require_per=False for jobs without test bodies, such as pretraining).
N_PROCS workers run in parallel.
"""
from procs import N_PROCS  # single-threaded BLAS workers; must precede numpy
import os
import json
import time
from multiprocessing import Pool
import numpy as np


def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x)}")


def _hashable(v):
    return tuple(_hashable(x) for x in v) if isinstance(v, (list, tuple)) else v


def key_of(row, key):
    """job key; lists and tuples compare equal, since JSON stores a tuple as a list."""
    return tuple(_hashable(row.get(k)) for k in key)


def load_rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def _call(args):
    work, job, require_per = args
    t0 = time.time()
    rec = {**work(job), **job}
    rec.setdefault("sec", round(time.time() - t0, 1))
    if require_per and not any(k.startswith("per_") and isinstance(v, (list, np.ndarray)) for k, v in rec.items()):
        raise ValueError(f"job {job} returned no per-body field (per_*)")
    return rec


def run(jobs, work, out, key, procs=None, require_per=True):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = {key_of(r, key) for r in load_rows(out)}
    todo = [j for j in jobs if key_of(j, key) not in done]
    if len({key_of(j, key) for j in todo}) != len(todo):
        raise ValueError("duplicate job keys")
    print(f"{out}: {len(todo)} jobs to run ({len(jobs) - len(todo)} of {len(jobs)} already done)", flush=True)
    if not todo:
        return
    with Pool(procs or N_PROCS) as pool, open(out, "a") as f:
        for rec in pool.imap_unordered(_call, [(work, j, require_per) for j in todo]):
            f.write(json.dumps(rec, default=_jsonable) + "\n")
            f.flush()
            brief = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()
                     if not isinstance(v, (list, dict))}
            print(" ".join(f"{k}={v}" for k, v in brief.items()), flush=True)
