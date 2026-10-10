"""Чому ImpactNet гірша в shift: чи P_in у DT-режимі містить α-нагрів, якого нема в проксі (нагрів + омічна)?
Порівнюємо істинний p_input_MW з проксі і з нейтронами (∝ P_fus) у звичайному та shift-режимах."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drivers as D  # noqa: E402

base = Path("~/risk_data").expanduser()
cols = ["shot_id", "mag_ip_MA", "mag_vloop_V", "nbi_power_MW", "ich_power_MW", "ech_power_MW", "neutron_total",
        "p_input_MW", "p_rad_MW", "ip_MA"] + D.BOLO
for name in ("r501", "rs601"):
    m = pd.read_csv(base / name / "measured.csv", usecols=cols)
    m = m[m.ip_MA > 1.0].ffill()
    proxy = (m.nbi_power_MW.fillna(0) + m.ich_power_MW.fillna(0) + m.ech_power_MW.fillna(0)
             + (m.mag_vloop_V * m.mag_ip_MA).clip(lower=0))
    resid = m.p_input_MW - proxy
    c = np.corrcoef(resid, m.neutron_total)[0, 1]
    k = np.linalg.lstsq(m[["neutron_total"]].to_numpy(), resid.to_numpy(), rcond=None)[0][0]
    print(f"{name}: P_in істинна {m.p_input_MW.median():.2f} МВт, проксі {proxy.median():.2f}; "
          f"залишок медіана {resid.median():+.2f}, 90% {resid.quantile(.9):+.2f}; corr(залишок, нейтрони) {c:.3f}, k={k:.3g}")
    frad_t = np.where(m.p_input_MW > 1, m.p_rad_MW / m.p_input_MW.clip(lower=1e-6), 0)
    frad_p = np.where(proxy > 1, sum(m[b] for b in D.BOLO) / proxy.clip(lower=1e-6), 0)
    s = frad_t > 0.3
    print(f"      frad (де істинна > 0,3): проксі / істинна — медіана {np.median(frad_p[s] / frad_t[s]):.3f}")
