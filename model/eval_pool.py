"""Об'єднана оцінка кількох прогонів заліку: метрика по всіх розрядах разом.

    python model/eval_pool.py v2: runs/v2_s1_data runs/v2_s20261001_data -- v3: runs/v3_s1_data ...

Один прогін = ~55 зривів → шум ±5–25 пунктів. Об'єднання 3–4 прогонів (різні
сиди) дає 160–220 зривів і відрізняє моделі надійніше. Поріг підбирається по
всіх тихих розрядах разом (як якби залік був на одному великому наборі).
Бутстреп по розрядах — 90 % інтервал для об'єму і TPR@5 %.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_run import shots_from  # noqa: E402
from metric import LEADS, report  # noqa: E402


def load_runs(dirs):
    shots = []
    for d in dirs:
        m = pd.read_csv(f"{d}/measured.csv", usecols=["shot_id", "t", "p"])
        s = pd.read_csv(f"{d}/shots.csv")
        shots += shots_from(m, s)
    return shots


def boot(shots, n=200, seed=0):
    rng = np.random.default_rng(seed)
    vols, tprs = [], {l: [] for l in LEADS}
    for _ in range(n):
        idx = rng.integers(0, len(shots), len(shots))
        r = report([shots[i] for i in idx])
        vols.append(r["volume"])
        for l in LEADS:
            tprs[l].append(r[l]["tpr_at_budget"])
    q = lambda a: (np.percentile(a, 5), np.percentile(a, 95))  # noqa: E731
    return q(vols), {l: q(tprs[l]) for l in LEADS}


if __name__ == "__main__":
    groups, cur = {}, None
    for a in sys.argv[1:]:
        if a == "--":
            continue
        if a.endswith(":"):
            cur = a[:-1]
            groups[cur] = []
        else:
            groups[cur].append(a)
    print(f"{'модель':10s} {'зривів':>6s} {'об’єм':>14s} | TPR@5%: " + "  ".join(f"{l:>12.2f}с" for l in LEADS))
    for name, dirs in groups.items():
        shots = load_runs(dirs)
        r = report(shots)
        (vlo, vhi), ci = boot(shots)
        print(f"{name:10s} {r['n_dis']:6d} {r['volume']:.3f} [{vlo:.3f},{vhi:.3f}] | "
              + "  ".join(f"{r[l]['tpr_at_budget']:.3f}[{ci[l][0]:.2f},{ci[l][1]:.2f}]" for l in LEADS))
