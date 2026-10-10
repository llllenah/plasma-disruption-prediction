"""Калібрування проксі на кількох розрядах + звірка offline/онлайн + наскільки драйвери з приладів
збігаються з істинними (кореляція на плато)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as D  # noqa: E402

m = pd.read_csv(Path("~/risk_data/r501/measured.csv").expanduser())
shots = [g.reset_index(drop=True) for _, g in m.groupby("shot_id", sort=False)]
calib = D.fit_calibration(shots[:30])
print("калібрування:", {k: round(v, 4) for k, v in calib.items()})
print("offline == онлайн, макс. відносна розбіжність:", f"{D.test_equal(shots[0], calib):.1e}")
rows = []
for g in shots[30:60]:
    X, drv = D.offline(g, calib)
    tr = D.truth_drivers(g)
    ip = g.ip_MA.clip(lower=0.05).to_numpy()
    true = dict(fgw=D.offline.__globals__["pd"].Series(g.ne_bar_1e20 / (ip / D.GW_AREA)).to_numpy(),
                q95=tr["q95"], frad=np.where(g.p_input_MW > 1, g.p_rad_MW / g.p_input_MW.clip(lower=1e-6), 0),
                beta_n=D.BETA_C * g.w_th_MJ / (ip * g.bt_T))
    ok = (g.ip_MA > 1.0).to_numpy()
    rows.append(pd.DataFrame({**{f"p_{k}": drv[ok, i] for i, k in enumerate(true)}, **{f"t_{k}": np.asarray(v)[ok] for k, v in true.items()}}))
r = pd.concat(rows)
for k in ("fgw", "q95", "frad", "beta_n"):
    a, b = r[f"p_{k}"], r[f"t_{k}"]
    s = np.isfinite(a) & np.isfinite(b)
    print(f"{k:7s}: corr {np.corrcoef(a[s], b[s])[0, 1]:.3f}, медіана |Δ|/x {np.median(np.abs(a[s] - b[s]) / np.abs(b[s]).clip(1e-3)):.3f}")
