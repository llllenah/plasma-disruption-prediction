"""Перевірка: атрибуція fusion-sim береться в момент максимуму λ моделі по ВСЬОМУ розряду, включно з прогрівом
на старті (t ≈ 0, струму ще нема). Беремо ті самі онлайн-виходи моделі з measured.csv, але максимум λ шукаємо
лише там, де плазма є (струм > порогу симулятора, не запрограмований спад) і до гасіння.

    python work/fs_moment.py measured.csv shots.csv run.json
"""
import json
import sys

import numpy as np
import pandas as pd

M = ["greenwald", "q95", "radiation", "troyon", "other"]
IP_MIN = 0.3   # перевіряється нижче з targets.ph.IP_MIN
try:
    sys.path.insert(0, "risk_model")
    import targets as T
    IP_MIN = T.ph.IP_MIN
except Exception:
    pass
sh = pd.read_csv(sys.argv[2]).set_index("shot_id")
run = json.load(open(sys.argv[3], encoding="utf-8"))
true = {s["shot_id"]: s["true_cause"]["dominant"] for s in run["shots"] if s.get("true_cause")}
fs = {s["shot_id"]: s["predicted_cause"]["dominant"] for s in run["shots"] if s.get("true_cause")}
cols = ["shot_id", "t", "mag_ip_MA", "prog_ip_MA", "model_lambda"] + [f"share_{m}" for m in M]
res = {}
first = {}
for ch in pd.read_csv(sys.argv[1], usecols=cols, chunksize=300_000):
    for sid, g in ch.groupby("shot_id"):
        res.setdefault(sid, []).append(g)
rows = []
for sid in sorted(true):
    g = pd.concat(res[sid]).sort_values("t")
    r = sh.loc[sid]
    lam = g.model_lambda.to_numpy()
    t = g.t.to_numpy()
    pk = g.prog_ip_MA.cummax().to_numpy()
    ok = (g.mag_ip_MA.to_numpy() > IP_MIN) & ~(g.prog_ip_MA.to_numpy() < 0.9 * pk) & (t < r.t_tq_s)
    S = g[[f"share_{m}" for m in M]].to_numpy()
    i_all = int(np.argmax(np.where(t < r.t_tq_s, lam, -1)))
    i_ok = np.where(ok)[0][np.argmax(lam[ok])]
    rows.append(dict(sid=sid, true=true[sid], fs=fs[sid], all_t=t[i_all], all=M[S[i_all].argmax()], ok_t=t[i_ok], ok=M[S[i_ok].argmax()]))
d = pd.DataFrame(rows)
print("IP_MIN", IP_MIN)
print("як у fusion-sim (fs):", (d.fs == d.true).sum(), "з", len(d))
print("максимум λ по всьому розряду (наше відтворення):", (d["all"] == d.true).sum(), "| збіг з fs", (d["all"] == d.fs).mean().round(2))
print("максимум λ лише з плазмою:", (d.ok == d.true).sum(), "з", len(d))
for m in M[:3]:
    k = d.true == m
    print(f"  {m}: fs {(d.fs[k] == m).sum()}/{k.sum()}  виправлено {(d.ok[k] == m).sum()}/{k.sum()}")
print("моменти, де fs помиляється:")
print(d[d.fs != d.true][["sid", "true", "fs", "all_t", "ok_t", "ok"]].to_string(index=False))
d.to_csv("work/fs_moment.csv", index=False)
