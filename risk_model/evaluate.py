"""Перевірка на відкладених розрядах (test: r508 + rs605) — за рядками і за фактом зриву (shots.csv).

    python risk_model/evaluate.py [TAG]

1. Рядки: λ, ризики механізмів, частки — по режимах (звичайний / shift) і рівнях ризику.
2. Постфактум: зрив у симуляторі — пуассонівський «кубик» з інтенсивністю λ. Якщо λ правильна,
   накопичений ризик до запуску зриву Λ = ∫λ dt ~ Exp(1) (середнє 1, медіана 0,69), а сума
   1 − e^(−Λ) по розрядах ≈ кількості зривів. Перевіряємо це для λ_true і для λ мережі.
   Плюс пуассонівська log-правдоподібність розрядів: мережа проти «справжньої» λ і проти сталої.
3. Механізм у момент зриву: домінуючий за мережею проти справжнього.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import targets as T  # noqa: E402
from train import DATA, metrics  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
z = np.load(DATA / f"pred_test{TAG}.npz", allow_pickle=True)
zp, risk_t, shot, t, src = z["z"], z["risk"], z["shot"], z["t"], z["src"]
DTR = 0.025                                  # крок рядків (кожен 5-й крок по 5 мс)

print("== 1. рядки ==")
for name, msk in (("звичайний", ~np.char.startswith(src.astype(str), "rs")), ("shift", np.char.startswith(src.astype(str), "rs"))):
    m = metrics(zp[msk], risk_t[msk])
    print(f"{name:10s}: λ MAE {m['lam_mae']:.4f}, R² {m['lam_r2']:.3f}, corr(log) {m['lam_logcorr']:.3f}, AUC(λ>0,1) {m['auc_lam>0.1']:.3f} | "
          f"частки: MAE {m['share_mae_pp@0.01']:.1f} п.п. (λ>0,01), {m['share_mae_pp@0.1']:.1f} п.п. (λ>0,1); "
          f"домінуючий: {m['dom_acc@0.01']:.3f} / {m['dom_acc@0.1']:.3f}")
    print("           R² ризиків:", {k[3:]: round(v, 3) for k, v in m.items() if k.startswith("r2_")})

C_t, share_t, lam_t = T.decompose(risk_t)
C_p = np.exp(zp)
share_p = 100 * C_p / C_p.sum(1, keepdims=True)
lam_p = 1 - np.exp(-C_p.sum(1))
risk_p = 1 - np.exp(-C_p)
# симулятор додає фонову «підлогу» 0,005 + r_lm (вона ж дає спонтанні зриви) — однаково для обох
lam_t_eff = np.maximum(lam_t, 0.005 + np.minimum(risk_t[:, 4], 0.01))
lam_p_eff = np.maximum(lam_p, 0.005 + np.minimum(risk_p[:, 4], 0.01))

print("\n== 2. постфактум: зрив / не зрив (shots.csv) ==")
rows = []
for s in np.unique(src):
    sh = pd.read_csv(DATA / f"shots_{s}.csv").set_index("shot_id")
    for sid in sh.index:
        k = (src == s) & (shot == sid)
        if not k.any():
            continue
        r = sh.loc[sid]
        dis = bool(r.disrupted) and r.t_trigger_s == r.t_trigger_s
        j = np.where(k)[0]
        last = j[-1]
        rows.append(dict(src=s, sid=sid, scen=r.scenario, dis=dis, spont=bool(r.spontaneous), wall=bool(r.wall_contact),
                         T=k.sum() * DTR, Lt=lam_t_eff[k].sum() * DTR, Lp=lam_p_eff[k].sum() * DTR,
                         lt_at=lam_t_eff[last], lp_at=lam_p_eff[last],
                         dom_t=T.MECH[share_t[last].argmax()] if dis else "", dom_p=T.MECH[share_p[last].argmax()] if dis else "",
                         lam_true_at=lam_t[last]))
d = pd.DataFrame(rows)
dd = d[d.dis]
print(f"розрядів {len(d)}, зривів {len(dd)} (зокрема спонтанних {int(dd.spont.sum())})")
for c, nm in (("Lt", "справжня λ"), ("Lp", "λ мережі")):
    print(f"  {nm:11s}: Λ до зриву — середнє {dd[c].mean():.2f}, медіана {dd[c].median():.2f} (Exp(1): 1 / 0,69);"
          f"  очікувано зривів Σ(1−e^(−Λ)) = {(1 - np.exp(-d[c])).sum():.1f} при фактичних {len(dd)}")
const = len(dd) / d["T"].sum()                     # стала інтенсивність, підібрана постфактум
for nm, L, la in (("справжня λ", d.Lt, d.lt_at), ("λ мережі", d.Lp, d.lp_at),
                  ("стала λ", d["T"] * const, pd.Series(const, index=d.index))):
    ll = (np.log(la[d.dis]).sum() - L.sum()) / len(d)
    print(f"  log-правдоподібність на розряд: {nm:11s} {ll:+.3f}")

print("\n== 3. механізм у момент запуску зриву (не спонтанні, справжня λ > 0,01) ==")
x = dd[~dd.spont & (dd.lam_true_at > 0.01)]
print(f"  збіг домінуючого механізму: {(x.dom_t == x.dom_p).mean():.3f} ({len(x)} зривів)")
print(pd.crosstab(x.dom_t, x.dom_p, rownames=["справжній"], colnames=["мережа"]).to_string())
print("\n  за сценарієм (мережа):")
print(pd.crosstab(x.scen, x.dom_p).to_string())
d.to_csv(DATA / f"eval_shots{TAG}.csv", index=False)
