"""Process-pool settings shared by every script.  Import this BEFORE numpy.

N_PROCS (environment variable, default 2) sets the number of worker processes; each worker is
single-threaded, since BLAS threads only oversubscribe the cores when several workers run.
"""
import os

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
N_PROCS = int(os.environ.get("N_PROCS", "2"))
