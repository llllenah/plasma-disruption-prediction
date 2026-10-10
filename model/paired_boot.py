"""Парний бутстреп різниці двох моделей на тих самих розрядах (ті самі сиди заліку).

    python model/paired_boot.py v5 v2 s1 shift_s1 s20261001 s42
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pool import load_runs  # noqa: E402
from metric import LEADS, report  # noqa: E402

a_name, b_name, *sets = sys.argv[1:]
A = load_runs([f"runs/{a_name}_{s}_data" for s in sets])
B = load_runs([f"runs/{b_name}_{s}_data" for s in sets])
assert len(A) == len(B)
rng = np.random.default_rng(0)
d_vol, d_tpr = [], {l: [] for l in LEADS}
for _ in range(300):
    idx = rng.integers(0, len(A), len(A))
    ra, rb = report([A[i] for i in idx]), report([B[i] for i in idx])
    d_vol.append(ra["volume"] - rb["volume"])
    for l in LEADS:
        d_tpr[l].append(ra[l]["tpr_at_budget"] - rb[l]["tpr_at_budget"])
q = lambda x: f"{np.mean(x):+.3f} [{np.percentile(x, 5):+.3f}, {np.percentile(x, 95):+.3f}]  P(>0)={np.mean(np.array(x) > 0):.2f}"  # noqa: E731
print(f"{a_name} − {b_name}, розрядів {len(A)}")
print("  об'єм     ", q(d_vol))
for l in LEADS:
    print(f"  TPR@5% {l:4.2f}с", q(d_tpr[l]))
