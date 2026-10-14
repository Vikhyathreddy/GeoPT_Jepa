"""Follow-up: is the prompt a necessary 'mode selector'?  Task A in the flow frame carries no
information in its prompt (it is the same constant for every sample), so we can set it to zero:
the lifted model then runs in its static (no-dynamics) mode."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import numpy as np
import finetune as F

_orig = F.prompts
F.prompts = lambda Dd, task, ids, shift, Q: np.zeros_like(_orig(Dd, task, ids, shift, Q))
D = F.load()
with open("results/followup_zero_prompt.jsonl", "w") as f:
    for v in ["L", "S"]:
        for s in [0, 1]:
            r, _, _ = F.finetune(D, v, "A", 64, s)
            r["prompt"] = "zero"
            f.write(json.dumps(r) + "\n")
            print(v, "A zero-prompt seed", s, "relL2 %.4f" % r["rel_l2"], flush=True)
