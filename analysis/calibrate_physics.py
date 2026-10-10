"""Калибровка физики на jet_dd_s1 и проверка восстановленной λ против disruption_risk.

    python analysis/calibrate_physics.py data/jet_dd_s1

Пишет analysis/physics_coef.json (коэффициенты q95 по форме).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics as ph  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else "data/jet_dd_s1"
ts = pd.read_csv(f"{SRC}/timeseries.csv")

# ── q95 по форме: log(q95·Ip/Bt) = c0 + c1·κ² + c2·δ ──
d = ts[(ts.ip_MA > 1.0) & (ts.w_th_MJ > 0.05) & ~ts.disrupted]
kappa, delta = ph.shape(d.xpoint_r_m, d.xpoint_z_m, d.axis_z_m)
M = np.c_[np.ones(len(d)), kappa ** 2, delta]
coef, *_ = np.linalg.lstsq(M, np.log(d.q95 * d.ip_MA / d.bt_T).to_numpy(), rcond=None)
q_hat = ph.q95_from_shape(d.bt_T, d.ip_MA, d.xpoint_r_m, d.xpoint_z_m, d.axis_r_m, d.axis_z_m, coef)
print("q95: медианная отн. ошибка", np.median(np.abs(q_hat - d.q95) / d.q95).round(4))

# ── λ из истинных величин против disruption_risk ──
rows = []
for sid, g in ts.groupby("shot_id"):
    dt = np.diff(g.time_s.to_numpy(), prepend=g.time_s.iloc[0] - 0.02).clip(1e-3)
    f_gw = g.ne_bar_1e20 / (g.ip_MA.clip(lower=0.05) / ph.GW_AREA)
    beta_n = ph.BETA_C * g.w_th_MJ / (g.ip_MA.clip(lower=0.05) * g.bt_T)
    q95 = ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m, g.axis_z_m, coef)
    lam, _ = ph.hazard(f_gw, beta_n, q95, g.p_rad_MW, g.p_input_MW, g.ip_MA, dt[0])
    lam_q, _ = ph.hazard(f_gw, beta_n, g.q95, g.p_rad_MW, g.p_input_MW, g.ip_MA, dt[0])
    pk = g.prog_ip_MA.max()
    ok = (g.prog_ip_MA >= 0.95 * pk) & ~g.disrupted & (g.w_th_MJ > 0.05)
    rows.append(pd.DataFrame(dict(true=g.disruption_risk[ok], hat=lam[ok.to_numpy()], hat_q=lam_q[ok.to_numpy()])))
r = pd.concat(rows)
for c in ("hat", "hat_q"):
    lt, lh = np.log(r.true), np.log(r[c].clip(lower=1e-6))
    print(f"λ ({'q95 по форме' if c == 'hat' else 'истинный q95'}): corr(log)={np.corrcoef(lt, lh)[0, 1]:.4f}  "
          f"медиана |Δλ|/λ={np.median(np.abs(r[c] - r.true) / r.true):.4f}")

Path("analysis/physics_coef.json").write_text(json.dumps(dict(q95_coef=list(map(float, coef))), indent=1))
print("коэффициенты q95:", np.round(coef, 4))
