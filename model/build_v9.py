"""features_v8/*.parquet + онлайн-ознаки нейромережі → features_v9/*.parquet.

    python model/build_v9.py

Ознаки мережі рахуються по повному розряду з кроком 5 мс (riskfeat.offline) і беруться в ті самі
моменти, що й рядки v8. Перевірка: offline == онлайн-клас RiskNetOnline на першому розряді.
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
import riskfeat as R  # noqa: E402

W = R.load()
CH = [str(c) for c in W["channels"]]
out_dir = ROOT / "features_v9"
out_dir.mkdir(exist_ok=True)
checked = False
for f in sorted(glob.glob(str(ROOT / "features_v8" / "tr*.parquet"))):
    src = Path(f).stem
    dst = out_dir / f"{src}.parquet"
    if dst.exists():
        print(src, "уже є")
        continue
    v8 = pd.read_parquet(f)
    m = pd.read_csv(ROOT / "datasets" / src / "measured.csv", usecols=["shot_id", "t"] + CH)
    parts = []
    for sid, g in m.groupby("shot_id", sort=False):
        g = g.reset_index(drop=True)
        F = R.offline(W, g)
        if not checked:                              # offline == онлайн (перші 600 кроків)
            on = R.RiskNetOnline(W)
            ref = np.array([on.step({c: g.at[i, c] for c in CH}) for i in range(600)])
            err = np.nanmax(np.abs(ref - F[:600]))
            print(f"звірка offline/онлайн: макс. розбіжність {err:.1e}")
            assert err < 1e-6
            checked = True
        d = pd.DataFrame(F, columns=R.NAMES)
        d["shot_id"] = sid
        d["_k"] = np.round(g.t.to_numpy() / R.DT).astype(np.int64)
        parts.append(d)
    nn = pd.concat(parts, ignore_index=True)
    v8["_k"] = np.round(v8.time.to_numpy() / R.DT).astype(np.int64)
    v9 = v8.merge(nn, on=["shot_id", "_k"], how="left").drop(columns="_k")
    assert len(v9) == len(v8) and v9.nn_lam.notna().mean() > 0.99, "рядки не зійшлися"
    v9.to_parquet(dst)
    print(f"{src}: {len(v9)} рядків, nn_lam: медіана {v9.nn_lam.median():.4f}, 99% {v9.nn_lam.quantile(.99):.3f}", flush=True)
