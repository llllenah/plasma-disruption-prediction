"""ROC AUC на рівні розрядів (як у заліку) для кожної моделі з офіційних звітів toksym (staircase = бігучий максимум p).
Розряд-зрив «спійманий» на рівні порогу θ, якщо max p до (t_TQ − lead) ≥ θ; тихий — «хибний», якщо max p ≥ θ.
AUC = P(рівень зриву > рівень тихого) + ½·P(рівні). Об'єм = середнє AUC по 5 запасах — звірка з eval_pool."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SETS = ("s1", "shift_s1", "s20261001", "s42")
MODELS = {"бейзлайн": ("runs", "baseline", {"s1": "baseline_s1", "shift_s1": "base_shift_s1"}),
          "v6": ("runs", "v6", None), "v6b": ("runs", "v6b", None), "v8": ("runs", "v8", None),
          "nocs": ("runs", "nocs", None), "clean": ("runs", "clean", None),
          "clean_big": ("pod_results/runs_pod2", "clean_big", None), "clean_9k": ("pod_results/runs_pod2", "clean_9k", None),
          "ens5": ("pod_results/runs_pod2", "clean_9k_ens5", None)}
LEADS = (0.01, 0.03, 0.1, 0.3, 1.0)


def levels(rep, lead):
    dis, quiet = [], []
    for s in rep["shots"]:
        if s["spontaneous"] or s["wall_contact"]:
            continue
        st = np.array(s["staircase"]) if s["staircase"] else np.zeros((1, 2))
        if s["disrupted"]:
            ok = st[:, 0] <= s["t_tq"] - lead + 1e-9
            dis.append(st[ok, 1].max() if ok.any() else 0.0)
        else:
            quiet.append(st[:, 1].max())
    return dis, quiet


def auc(d, q):
    d, q = np.asarray(d)[:, None], np.asarray(q)[None, :]
    return float((d > q).mean() + 0.5 * (d == q).mean())


rows = {}
for name, (folder, prefix, special) in MODELS.items():
    reps = []
    for s in SETS:
        f = ROOT / folder / f"{(special or {}).get(s, f'{prefix}_{s}')}.json"
        if special is not None and s not in special:
            continue
        if f.exists():
            reps.append(json.loads(f.read_text()))
    if not reps:
        continue
    res = {}
    for L in LEADS:
        d, q = [], []
        for r in reps:
            a, b = levels(r, L)
            d += a
            q += b
        res[L] = auc(d, q)
    res["volume"] = float(np.mean([res[L] for L in LEADS]))
    res["n_sets"] = len(reps)
    rows[name] = res
    print(f"{name:10s} наборів {len(reps)} | AUC 30 мс {res[0.03]:.3f} | 0,1 с {res[0.1]:.3f} | 0,3 с {res[0.3]:.3f} | 1 с {res[1.0]:.3f} | об'єм {res['volume']:.3f}")
(ROOT / "presentation" / "auc_table.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
