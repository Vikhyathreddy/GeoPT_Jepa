"""Follow-up: does prompt misalignment hurt once fine-tuning can no longer re-map the prompt?

The first query-embedding layer (q.1, which reads [position features, prompt v, |v|]) is frozen at
its pretrained/initial values, so a rotated prompt cannot be undone by re-learning that layer.
"""
import os
from procs import N_PROCS  # single-threaded BLAS workers; must precede numpy
import json
import numpy as np
from multiprocessing import Pool


def work(j):
    import finetune as F
    D = F.load()
    r, _, _ = F.finetune(D, j["variant"], j["task"], 64, j["seed"], shift=np.radians(j["shift"]), freeze=("q.1",))
    r["freeze"] = "q.1"
    return r


if __name__ == "__main__":
    jobs = [dict(variant=v, task=t, shift=s, seed=sd) for v in ["L", "S"] for t in "AB" for s in [0, 180] for sd in [0, 1]]
    with Pool(N_PROCS) as p, open("results/followup_freeze.jsonl", "w") as f:
        for r in p.imap_unordered(work, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["variant"], r["task"], "shift", r["shift"], "seed", r["seed"], "relL2 %.4f" % r["rel_l2"], flush=True)
