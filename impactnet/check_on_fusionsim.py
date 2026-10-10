"""Перевірка ImpactNet на симуляторі з діями (fusion_sim_client) проти його справжнього ризику.

    python check_on_fusionsim.py <шлях до теки з tools/> [кількість розрядів] [сид]

1. Чи збігаються назви 100 каналів симулятора з channels.json мережі.
2. На кожному кроці: ImpactNet проти disruption_risk і risk_* симулятора
   (risk_* у відповіді — сигмоїди ДО куба; мережа видає rᵢ³, а для locked mode — r_lm).
3. Зведення: R² λ, збіг головного механізму там, де ризик помітний (λ > 0,1).
Без дій — чистий розряд. Щоб перевірити стійкість до дій, розкоментуйте блок «дія» нижче.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(sys.argv[1]) / "tools"))
from fusion_sim_client import FusionSim  # noqa: E402
from impactnet import MECH, ImpactStep  # noqa: E402

N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ours = json.loads((HERE / "channels.json").read_text())
SIM_KEYS = {"greenwald": ("risk_greenwald", 3), "q95": ("risk_q95", 3), "radiation": ("risk_radiation", 3),
            "troyon": ("risk_beta", 3), "other": ("risk_locked_mode", 1)}
lam_p, lam_t, dom_p, dom_t = [], [], [], []
for shot in range(N):
    st = ImpactStep()
    with FusionSim("jet", f"generated:{SEED}:{shot}") as sim:
        names = sim.program()["channel_names"]
        if shot == 0:
            miss, extra = set(ours) - set(names), set(names) - set(ours)
            print(f"канали: у симуляторі {len(names)}; бракує для мережі {sorted(miss) or 'нічого'}; "
                  f"зайві {sorted(extra) or 'нічого'}")
        while True:
            s = sim.step()
            if s["status"] != "Running" or s["t_tq"] is not None or s["t_trigger"] is not None:
                break
            o = st.step(dict(zip(names, s["channels"])))
            # дія (для перевірки стійкості):
            # if s["f_greenwald"] > 0.9: sim.set("ne_target", 0.8 * s["prog_ne"])
            true_r = {k: (s[f] or 0.0) ** p for k, (f, p) in SIM_KEYS.items()}
            lam_p.append(o["risk_total"])
            lam_t.append(s["disruption_risk"])
            if s["disruption_risk"] > 0.1:
                dom_p.append(max(o["share_pct"], key=o["share_pct"].get))
                dom_t.append(max(true_r, key=true_r.get))
lp, lt = np.array(lam_p), np.array(lam_t)
r2 = 1 - ((lp - lt) ** 2).sum() / ((lt - lt.mean()) ** 2).sum()
print(f"{N} розрядів, {len(lt)} кроків: R² λ = {r2:.3f} (λ симулятора має нижню межу 0,005 + r_lm, мережа — ні; "
      f"MAE {np.abs(lp - lt).mean():.4f})")
if dom_t:
    print(f"головний механізм при λ > 0,1: збіг {np.mean(np.array(dom_p) == np.array(dom_t)):.1%} на {len(dom_t)} кроках")
else:
    print("кроків з λ > 0,1 не було — візьміть більше розрядів")
