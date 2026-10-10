"""Чи справді зрив — пуассонівський «кубик» з інтенсивністю λ (як у методичці)?

    python analysis/trigger_law.py datasets/oracle_s1 datasets/v3_301 ...

Якщо так, накопичений ризик до моменту зриву Λ = ∫₀^t_trigger λ dt ~ Exp(1):
середнє 1, медіана 0,69, P(Λ < 0,1) ≈ 10 %. Якщо зриви стаються «по порогу»
або за розкладом сценарію — розподіл Λ буде зовсім іншим.
Також: λ у момент зриву проти максимуму λ у тихих розрядах, і момент зриву
відносно початку плато / сценарію.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import physics as ph  # noqa: E402

Q = json.loads((Path(__file__).resolve().parent / "physics_coef.json").read_text())["q95_coef"]
cols = ["shot_id", "t", "prog_ip_MA", "prog_ne_1e20", "ip_MA", "bt_T", "ne_bar_1e20", "w_th_MJ", "p_rad_MW",
        "p_input_MW", "axis_r_m", "axis_z_m", "xpoint_r_m", "xpoint_z_m"]
rows = []
for src in sys.argv[1:]:
    m = pd.read_csv(f"{src}/measured.csv", usecols=cols)
    sh = pd.read_csv(f"{src}/shots.csv").set_index("shot_id")
    for sid, g in m.groupby("shot_id"):
        r = sh.loc[sid]
        ip = g.ip_MA.clip(lower=0.05)
        lam, terms = ph.hazard(g.ne_bar_1e20 / (ip / ph.GW_AREA), ph.BETA_C * g.w_th_MJ / (ip * g.bt_T),
                               ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m, g.axis_z_m, Q),
                               g.p_rad_MW, g.p_input_MW, g.ip_MA, 0.005)
        pk = g.prog_ip_MA.cummax()
        lam = np.where(g.prog_ip_MA < 0.9 * pk, 0.0, lam)
        t = g.t.to_numpy()
        flat = g.prog_ip_MA.to_numpy() >= 0.95 * g.prog_ip_MA.max()
        t_flat = t[flat][0] if flat.any() else np.nan
        if r.disrupted and r.t_trigger_s == r.t_trigger_s:
            k = t < r.t_trigger_s
            i = max(k.sum() - 1, 0)
            Lam = lam[k].sum() * 0.005
            rows.append(dict(src=src, sid=sid, kind="spont" if r.spontaneous else "disr", scen=r.scenario,
                             Lam=Lam, lam_at=lam[i], lam_max=lam[k].max() if k.any() else 0,
                             t_trig=r.t_trigger_s, t_from_flat=r.t_trigger_s - t_flat,
                             cause=max(terms, key=lambda c: np.asarray(terms[c])[i])))
        else:
            rows.append(dict(src=src, sid=sid, kind="quiet", scen=r.scenario, Lam=lam.sum() * 0.005,
                             lam_at=np.nan, lam_max=lam.max(), t_trig=np.nan, t_from_flat=np.nan, cause=""))
d = pd.DataFrame(rows)
dis = d[d.kind == "disr"]
q = d[d.kind == "quiet"]
print(f"розрядів {len(d)}: зривів {len(dis)}, спонтанних {(d.kind == 'spont').sum()}, тихих {len(q)}\n")
print("Λ = ∫λ dt до зриву (пуассон ⇒ Exp(1): середнє 1, медіана 0,69, частка < 0,1 ≈ 0,10):")
print(f"  зривні: середнє {dis.Lam.mean():.3f}, медіана {dis.Lam.median():.3f}, частка < 0,1 = {(dis.Lam < 0.1).mean():.2f}, > 3 = {(dis.Lam > 3).mean():.2f}")
print(f"  тихі (весь розряд): середнє {q.Lam.mean():.3f}, медіана {q.Lam.median():.3f};"
      f"  очікувана частка зривів за пуассоном = {(1 - np.exp(-q.Lam)).mean():.2f}")
print(f"\nλ у момент зриву: медіана {dis.lam_at.median():.3f}, 10-й перц. {dis.lam_at.quantile(.1):.3f}"
      f"  | max λ у тихих: медіана {q.lam_max.median():.3f}, 90-й перц. {q.lam_max.quantile(.9):.3f}")
print("\nза сценарієм (зривні): медіана Λ, λ у момент зриву, час від початку плато")
print(dis.groupby("scen").agg(n=("Lam", "size"), Lam=("Lam", "median"), lam_at=("lam_at", "median"),
                              t_from_flat=("t_from_flat", "median"), t_from_flat_std=("t_from_flat", "std")).round(3).to_string())
print("\nпричина (найбільший rᵢ³ у момент зриву):", dis.cause.value_counts().to_dict())
d.to_csv("analysis/trigger_law.csv", index=False)
