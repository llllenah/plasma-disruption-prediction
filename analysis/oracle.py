"""Стеля: скільки зривів можна спіймати, знаючи справжній ризик симулятора.

    python analysis/oracle.py datasets/oracle_s1 runs/v2_s1_data runs/baseline_s1_data

Оракул 1 «знає λ»:        p(t) = 1 − exp(−λ_true(t)·1 с), на спаді струму 0.
Оракул 2 «λ + передвісник»: як 1, але з моменту t_trigger p = 1 (ідеальний детектор).
λ_true — формула методички по справжньому стану (analysis/physics.py, точність 0,02 %).
Порівняння з нашою моделлю і бейзлайном на тих самих 200 розрядах заліку (сід 1).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
import physics as ph  # noqa: E402
from metric import LEADS, report  # noqa: E402

SRC, *RUNS = sys.argv[1:]
Q = json.loads((Path(__file__).resolve().parent / "physics_coef.json").read_text())["q95_coef"]
cols = ["shot_id", "t", "prog_ip_MA", "ip_MA", "bt_T", "ne_bar_1e20", "w_th_MJ", "p_rad_MW", "p_input_MW",
        "axis_r_m", "axis_z_m", "xpoint_r_m", "xpoint_z_m"]
m = pd.read_csv(f"{SRC}/measured.csv", usecols=cols)
sh = pd.read_csv(f"{SRC}/shots.csv").set_index("shot_id")

oracle1, oracle2 = [], []
HS = (0.3, 1.0, 3.0)
future = {h: [] for h in HS}
for sid, g in m.groupby("shot_id"):
    ip = g.ip_MA.clip(lower=0.05)
    f_gw = g.ne_bar_1e20 / (ip / ph.GW_AREA)
    beta = ph.BETA_C * g.w_th_MJ / (ip * g.bt_T)
    q95 = ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m, g.axis_z_m, Q)
    lam, _ = ph.hazard(f_gw, beta, q95, g.p_rad_MW, g.p_input_MW, g.ip_MA, 0.005)
    pk = g.prog_ip_MA.cummax()
    lam = np.where(g.prog_ip_MA < 0.9 * pk, 0.0, lam)          # на спаді жереб не кидається
    p1 = 1 - np.exp(-lam * 1.0)
    r = sh.loc[sid]
    t = g.t.to_numpy()
    p2 = p1.copy()
    if r.disrupted and r.t_trigger_s == r.t_trigger_s:
        p2[t >= r.t_trigger_s] = 1.0
    if r.spontaneous or r.wall_contact:
        continue
    base = dict(t=t, t_tq=float(r.t_tq_s) if r.disrupted else np.inf, quiet=not bool(r.disrupted))
    oracle1.append({**base, "p": p1})
    oracle2.append({**base, "p": p2})
    # оракул майбутнього: знає λ на H секунд уперед (але не результат жеребу)
    cum = np.r_[0, np.cumsum(lam * 0.005)]
    for h in HS:
        k = int(round(h / 0.005))
        j = np.minimum(np.arange(len(lam)) + k, len(lam))
        pf = 1 - np.exp(-(cum[j] - cum[np.arange(len(lam))]))
        if r.disrupted and r.t_trigger_s == r.t_trigger_s:
            pf = np.where(t >= r.t_trigger_s, 1.0, pf)
        future[h].append({**base, "p": pf})


def run_shots(d):
    pm = pd.read_csv(f"{d}/measured.csv", usecols=["shot_id", "t", "p"])
    s = pd.read_csv(f"{d}/shots.csv").set_index("shot_id")
    out = []
    for sid, g in pm.groupby("shot_id"):
        r = s.loc[sid]
        if r.spontaneous or r.wall_contact:
            continue
        out.append(dict(t=g.t.to_numpy(), p=g.p.to_numpy(), t_tq=float(r.t_tq_s) if r.disrupted else np.inf,
                        quiet=not bool(r.disrupted)))
    return out


rows = {"оракул: знає λ": report(oracle1), "оракул: λ + передвісник": report(oracle2)}
for h in HS:
    rows[f"оракул: майбутнє λ {h:g} с"] = report(future[h])
for d in RUNS:
    rows[Path(d).name.replace("_data", "")] = report(run_shots(d))

print(f"{'':28s} {'об’єм':>6s} | TPR@5%: " + "  ".join(f"{l:>5.2f}с" for l in LEADS) + " | AUC: " + "  ".join(f"{l:>5.2f}с" for l in LEADS))
for name, r in rows.items():
    print(f"{name:28s} {r['volume']:6.3f} | " + "  ".join(f"{r[l]['tpr_at_budget']:6.3f}" for l in LEADS)
          + " |      " + "  ".join(f"{r[l]['auc']:6.3f}" for l in LEADS))
