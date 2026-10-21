"""Step 4: apply the pre-registered pretraining-seed rule (SEED_REPLICATION.md) to every claim.

For each claim and pretraining seed k (0-4): d_k = mean over the 100 test bodies of the paired
per-body difference or relative change (per-body values first averaged over that seed's
fine-tuning seeds).  Robust: direction holds in >= 4 of 5 seeds AND the across-seed t interval
(df = 4) excludes zero on the claimed side.  Single-seed otherwise; contradicted if the interval
excludes zero on the other side; incomplete while fewer than 5 seeds exist.

usage: python seeds.py      -> results/seed_replication.json / .md
"""
import json
import numpy as np
from scipy import stats
import bootstrap as BS
from runner import load_rows

SEEDS = (0, 1, 2, 3, 4)


def across(d, side):
    """side = +1 / -1 for directional claims, 0 for reported-only, None for null-type."""
    d = np.asarray([x for x in d if x is not None], float)
    n = len(d)
    out = dict(n_seeds=n, values=[float(x) for x in d])
    if n < 2:
        return dict(out, verdict="incomplete")
    m, sd = d.mean(), d.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    out.update(mean=float(m), lo=float(m - h), hi=float(m + h))
    if side in (+1, -1):
        out["n_in_direction"] = int((np.sign(d) == side).sum())
    if n < 5:
        out["verdict"] = "incomplete"
    elif side in (+1, -1):
        if out["n_in_direction"] >= 4 and (out["lo"] > 0 if side > 0 else out["hi"] < 0):
            out["verdict"] = "robust"
        elif (out["hi"] < 0 if side > 0 else out["lo"] > 0):
            out["verdict"] = "contradicted"
        else:
            out["verdict"] = "single-seed"
    else:
        out["verdict"] = "reported"
    return out


# ----------------------------------------------------------------------------- Experiment 1
def exp1_claims():
    import exp1
    R = exp1.rows()
    sel = lambda v, tag, t, k: [r for r in R if (r["variant"], r["tag"], r["task"], r["n"], r["pretrain_seed"])
                                == (v, tag, t, 64, k)]
    rel = lambda a, b: BS.paired(a, b)["mean"] if a and b else None
    C = []
    for t in "AB":
        specs = [("E1-a", "clean L beats clean LT: err(LT)/err(L) - 1", +1, lambda k: rel(sel("LT", "clean", t, k), sel("L", "clean", t, k))),
                 ("E1-b", "clean L beats N: err(N)/err(L) - 1", +1, lambda k: rel(sel("N", "clean", t, k), sel("L", "clean", t, k))),
                 ("E1-d", "label sigma 2 degrades L: err(L,s2)/err(L) - 1", +1, lambda k: rel(sel("L", "lblfrz2", t, k), sel("L", "clean", t, k))),
                 ("E1-e", "at sigma 2 L beats clean LT: err(LT)/err(L,s2) - 1", +1, lambda k: rel(sel("LT", "clean", t, k), sel("L", "lblfrz2", t, k)))]
        if t == "A":
            specs.append(("E1-f", "at sigma 4 L beats clean LT: err(LT)/err(L,s4) - 1", +1,
                          lambda k: rel(sel("LT", "clean", t, k), sel("L", "lblfrz4", t, k))))
        for cid, desc, side, f in specs:
            C.append(dict(id=cid, task=t, claim=desc, side=side, **across([f(k) for k in SEEDS], side)))
    return C


# ----------------------------------------------------------------------------- Experiment 2
def exp2_claims():
    R = load_rows("results/exp2_finetune.jsonl")
    name = lambda k: "L" if k == 0 else f"Ls{k}"
    C = []
    for n in (64, 16):
        sel = lambda v, t, sh, m="normal": [r for r in R if (r["variant"], r["task"], r["n"], r["shift"], r["prompt_mode"],
                                                             r["freeze"]) == (v, t, n, sh, m, "")]
        per = lambda rows: np.mean([r["per_shape"] for r in rows], 0)
        full = lambda *rs: all(len(x) >= 3 for x in rs)

        def diff(k, t, sh, m="normal"):
            a, b = sel(name(k), t, sh, m), sel(name(k), t, 0)
            return float((per(a) - per(b)).mean()) if full(a, b) else None

        def did(k, t, sh):
            a, b, sa, sb = sel(name(k), t, sh), sel(name(k), t, 0), sel("S", t, sh), sel("S", t, 0)
            return float(((per(a) - per(b)) - (per(sa) - per(sb))).mean()) if full(a, b, sa, sb) else None

        for t in "AB":
            C.append(dict(id="E2-a", task=f"{t} n={n}", claim="zero prompt hurts L: err(zero) - err(0)", side=+1,
                          **across([diff(k, t, 0, "zero") for k in SEEDS], +1)))
            C.append(dict(id="E2-d", task=f"{t} n={n}", claim="L loses more than scratch at 90: diff-in-diff", side=+1,
                          **across([did(k, t, 90) for k in SEEDS], +1)))
            C.append(dict(id="E2-g", task=f"{t} n={n}", claim="90 helps L (own delta): err(90) - err(0)", side=-1,
                          **across([diff(k, t, 90) for k in SEEDS], -1)))
        C.append(dict(id="E2-b", task=f"B n={n}", claim="reversed prompt beats aligned for L: err(180) - err(0)",
                      side=-1, **across([diff(k, "B", 180) for k in SEEDS], -1)))
        c = across([diff(k, "A", 180) for k in SEEDS], None)
        res = []
        for k in SEEDS:
            for sh in (0, 90, 180):
                e = [r["rel_l2"] for r in sel(name(k), "A", sh)]
                if len(e) >= 2:
                    res.append(np.asarray(e) - np.mean(e))
        if res:
            dof = sum(len(x) - 1 for x in res)
            se = float(np.sqrt(sum((x ** 2).sum() for x in res) / dof) * np.sqrt(2 / 3))
            c["se_seed"] = se
            if c.get("n_seeds", 0) >= 5:
                c["verdict"] = "holds" if (c["lo"] <= 0 <= c["hi"]) or (abs(c["lo"]) <= 2 * se and abs(c["hi"]) <= 2 * se) \
                    else "fails"
        C.append(dict(id="E2-c", task=f"A n={n}", claim="180 costs about nothing for L (null-type)", side=None, **c))
    return C


# ----------------------------------------------------------------------------- Experiment 3
def exp3_claims():
    Pr, R = load_rows("results/exp3_probe.jsonl"), load_rows("results/exp3_runs.jsonl")
    bb = lambda b, k: b if k == 0 else f"{b}@s{k}"

    def probe(t, k):
        a = [x for x in Pr if (x["backbone"], x["task"]) == (bb("L", k), t)]
        b = [x for x in Pr if (x["backbone"], x["task"]) == (bb("LT", k), t)]
        return float(np.mean(a[0]["dec_per_body_rho"]) - np.mean(b[0]["dec_per_body_rho"])) if a and b else None

    def r1(t, k):
        a = [x for x in R if (x["regime"], x["backbone"], x["task"], x["seed"]) == ("R1", bb("L", k), t, 0)]
        b = [x for x in R if (x["regime"], x["backbone"], x["task"], x["seed"]) == ("R1", bb("LT", k), t, 0)]
        return float(np.mean(a[0]["per_body_rho"]) - np.mean(b[0]["per_body_rho"])) if a and b else None

    C = []
    for t in "AB":
        C.append(dict(id="E3-a", task=t, claim="L ranks better than LT, decoder probe: rho(L) - rho(LT)", side=+1,
                      **across([probe(t, k) for k in SEEDS], +1)))
        e3b = across([r1(t, k) for k in SEEDS], +1)
        e3b["verdict"] = "single-seed (R1 not replicated by design, 2026-10-21)"
        C.append(dict(id="E3-b", task=t, claim="L ranks better than LT, R1 (ft seed 0): rho(L) - rho(LT)", side=+1, **e3b))
    return C


def q1_decider():
    """E1-c': clean LT beats clean N (decides the question-1 branch; SEED_REPLICATION.md 2026-10-21)."""
    import exp1
    R = exp1.rows()
    sel = lambda v, t, k: [r for r in R if (r["variant"], r["tag"], r["task"], r["n"], r["pretrain_seed"])
                           == (v, "clean", t, 64, k)]
    C = []
    for t in "AB":
        d = [BS.paired(sel("LT", t, k), sel("N", t, k))["mean"] if sel("LT", t, k) and sel("N", t, k) else None
             for k in SEEDS]
        C.append(dict(id="E1-c'", task=t, claim="clean LT beats clean N: err(LT)/err(N) - 1 (decides Q1 branch)",
                      side=-1, **across(d, -1)))
    both = all(c["verdict"] == "robust" for c in C)
    done = all(c["verdict"] != "incomplete" for c in C)
    branch = ("branch 1: run trajectory noise 0.6 / 1.0" if both else
              "branch 2: close Q1 as not testable with this latent design") if done else "undecided (incomplete)"
    return C, branch


def main():
    Q1, branch = q1_decider()
    C = Q1 + exp1_claims() + exp2_claims() + exp3_claims()
    L = ["| id | task | claim | per-seed values (s0..s4) | in direction | across-seed mean [95% t interval] | verdict |",
         "|---|---|---|---|---|---|---|"]
    for c in C:
        vals = ", ".join(f"{v:+.4f}" for v in c["values"])
        iv = f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if "mean" in c else "-"
        nd = f"{c['n_in_direction']}/{c['n_seeds']}" if "n_in_direction" in c else "-"
        extra = f" (2 SE_seed = {2 * c['se_seed']:.4f})" if "se_seed" in c else ""
        L.append(f"| {c['id']} | {c['task']} | {c['claim']} | {vals} | {nd} | {iv}{extra} | {c['verdict']} |")
    text = f"Question-1 branch (decided by E1-c', reported first): **{branch}**\n\n" + "\n".join(L)
    json.dump(dict(q1_branch=branch, claims=C), open("results/seed_replication.json", "w"), indent=1)
    open("results/seed_replication.md", "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
