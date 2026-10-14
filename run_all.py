"""Run the full experiment grid (resumable).  usage: python run_all.py [main|prompt|frozen|lt0|all]"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
import json
import numpy as np
from multiprocessing import Pool

OUT = "results/finetune.jsonl"
STEPS = int(os.environ.get("FT_STEPS", "1000"))


def jobs(which):
    J = []
    if which in ("main", "all"):
        for v in ["S", "N", "J", "L", "LJ", "LT"]:
            for t in ["A", "B"]:
                for n in [16, 64, 256]:
                    for s in [0, 1]:
                        J.append(dict(variant=v, task=t, n=n, seed=s, shift=0, frozen=False))
    if which in ("prompt", "all"):
        for v in ["S", "L", "LT"]:
            for t in ["A", "B"]:
                for sh in [90, 180]:
                    for s in [0, 1]:
                        J.append(dict(variant=v, task=t, n=64, seed=s, shift=sh, frozen=False))
    if which in ("frozen", "all"):
        for v in ["S", "N", "J", "L", "LJ", "LT"]:
            for t in ["A", "B"]:
                for s in [0, 1]:
                    J.append(dict(variant=v, task=t, n=64, seed=s, shift=0, frozen=True))
    if which in ("lt0", "all"):
        for t in ["A", "B"]:
            for s in [0, 1]:
                J.append(dict(variant="LT0", task=t, n=64, seed=s, shift=0, frozen=False))
    return J


def key(j):
    return (j["variant"], j["task"], j["n"], j["seed"], j["shift"], j["frozen"])


_D = None


def work(j):
    global _D
    import finetune as F
    if _D is None:
        _D = F.load()
    rec, per, _ = F.finetune(_D, j["variant"], j["task"], j["n"], j["seed"],
                             shift=np.radians(j["shift"]), frozen=j["frozen"], steps=STEPS)
    rec["per_shape"] = [round(float(x), 5) for x in per]
    return rec


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["variant"], r["task"], r["n"], r["seed"], r["shift"], r["frozen"]))
    todo = [j for j in jobs(which) if key(j) not in done]
    print(f"{len(todo)} jobs to run ({len(done)} already done)", flush=True)
    with Pool(2) as pool, open(OUT, "a") as f:
        for rec in pool.imap_unordered(work, todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{rec['variant']:4s} {rec['task']} n={rec['n']:3d} s={rec['seed']} shift={rec['shift']:3d} "
                  f"frozen={int(rec['frozen'])} relL2={rec['rel_l2']:.4f} ({rec['sec']}s)", flush=True)
