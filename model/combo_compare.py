"""Объединение голов: «шумное ИЛИ» 1 − (1 − A)(1 − w·B) против простого максимума max(A, w·B).

    FEATURES=features_nocs FEAT_DIR=features_nocs EVAL_ONLY='tr10[1-6]|trs20[1-3]' \
        python model/combo_compare.py oof_0.1_4.5_a2_nocs.npz
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from metric import LEADS  # noqa: E402

df = train.load()
zs = [np.load(train.FEAT_DIR / f) for f in sys.argv[1:]]     # голова A — з першого; B — середнє всіх
ev = train.eval_mask(df)
a = zs[0]["a"][ev]
b = np.mean([z["b"] for z in zs], axis=0)[ev]
df = df[ev]
shift = df.src.str.startswith("trs").to_numpy()
print(f"{'объединение':24s} {'режим':6s} | объём  | TPR@5%: " + "  ".join(f"{l:>5.2f}с" for l in LEADS))
for w in (0.3, 0.5, 0.7):
    for name, p in ((f"1−(1−A)(1−{w}B)", 1 - (1 - a) * (1 - w * b)), (f"max(A, {w}B)", np.maximum(a, w * b))):
        for rn, msk in (("звич", ~shift), ("shift", shift)):
            r = train.eval_shots(df[msk], p[msk])
            print(f"{name:24s} {rn:6s} | {r['volume']:.3f} | " + "  ".join(f"{r[l]['tpr_at_budget']:6.3f}" for l in LEADS), flush=True)
