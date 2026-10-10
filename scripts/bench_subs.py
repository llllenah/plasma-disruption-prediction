"""Чистий замір кроку здач на цій машині: ті самі кроки (3 розряди з runs/clean_s1_data), крок за кроком.

    python work/bench_subs.py submission_clean_big submission_clean_9k_ens5 submission_v8
"""
import gc
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / "runs/clean_s1_data/measured.csv")
chans = [c for c in m.columns if c not in ("shot_id", "t", "p")]
shots = [g for _, g in m.groupby("shot_id", sort=False)][:3]
for sub in sys.argv[1:]:
    sys.path.insert(0, str(ROOT / sub))
    for mod in ("model", "features", "features_v8_base", "riskfeat"):
        sys.modules.pop(mod, None)
    M = importlib.import_module("model")
    tt = []
    gc.disable()
    for g in shots:
        mdl = M.Model()
        mdl.start_shot({"shot_id": 0, "dt": 0.005, "duration": float(g.t.iloc[-1]), "scenario": ""})
        for t, v in zip(g.t.to_numpy(), g[chans].to_numpy(float)):
            t0 = time.perf_counter()
            mdl.predict(float(t), dict(zip(chans, v)))
            tt.append(time.perf_counter() - t0)
    gc.enable()
    sys.path.pop(0)
    a = np.array(tt[200:]) * 1e3
    print(f"{sub:26s} кроків {len(a)}: p50 {np.median(a):.2f} мс, p99 {np.percentile(a, 99):.2f} мс, "
          f"макс {a.max():.1f} мс, > 5 мс: {(a > 5).mean() * 100:.2f} %", flush=True)
