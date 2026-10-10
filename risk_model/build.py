"""measured.csv + shots.csv → входи мережі та мітки (по одному .npz на набір).

    python risk_model/build.py ~/risk_data r501 r502 ...

Входи — ТІЛЬКИ 100 каналів приладів із measured.csv (той самий склад, що бачить модель на заліку;
без shot_id і без абсолютного часу t). Для кожного каналу: відмова приладу (nan) → останнє
нормальне значення, далі три причинні EMA (30 мс, 100 мс, 300 мс) — лише минуле.
Мітки — targets.py по істинних колонках `--extras` (на вхід не йдуть).

Які рядки беруться (shots.csv — «постфактум»):
  • струм вище порогу симулятора (IP_MIN) і не запрограмований спад — там симулятор кидає «кубик»;
  • у зривних розрядах — лише до моменту запуску зриву t_trigger (далі зрив уже вирішено);
  • кожен 5-й крок (25 мс) — сусідні кроки майже однакові.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import targets as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# рівно 100 приладів, які симулятор віддає моделі на заліку (заголовок прогону `toksym score`)
CHANNELS = [c for c in json.loads((ROOT / "impactnet" / "channels.json").read_text())
            if c not in ("shot_id", "t", "p")]
assert len(CHANNELS) == 100
TAUS = (0.03, 0.1, 0.3)
DT = 0.005
STRIDE = 5


def feature_names():
    return [f"{c}__ema{int(tau * 1000)}" for tau in TAUS for c in CHANNELS]


def build(src):
    m = pd.read_csv(src / "measured.csv")
    sh = pd.read_csv(src / "shots.csv").set_index("shot_id")
    assert not set(T.TRUTH_COLS) & set(CHANNELS), "істинні колонки не можуть бути входами"
    out = {k: [] for k in ("X", "risk", "lam_sim", "shot", "t", "ramp_ok")}
    for sid, g in m.groupby("shot_id", sort=False):
        g = g.reset_index(drop=True)
        r = sh.loc[sid]
        risk = T.mechanism_risks(g, DT)
        # звірка з формулою симулятора (analysis/physics.hazard) — до підлоги й маски струму
        ip = g.ip_MA.clip(lower=0.05)
        lam_sim, _ = T.ph.hazard(g.ne_bar_1e20 / (ip / T.ph.GW_AREA), T.ph.BETA_C * g.w_th_MJ / (ip * g.bt_T),
                                 T.ph.q95_from_shape(g.bt_T, g.ip_MA, g.xpoint_r_m, g.xpoint_z_m, g.axis_r_m,
                                                     g.axis_z_m, T.Q95_COEF),
                                 g.p_rad_MW, g.p_input_MW, g.ip_MA, DT)
        _, _, lam = T.decompose(risk)
        on = g.ip_MA.to_numpy() > T.ph.IP_MIN
        floor = 0.005 + np.minimum(risk[:, 4], 0.01)
        chk = on & (lam > floor)
        assert np.allclose(lam[chk], lam_sim[chk], rtol=1e-6, atol=1e-9), "розклад не збігся з формулою"

        x = g[CHANNELS].ffill()
        feats = [x.ewm(alpha=1 - np.exp(-DT / tau), adjust=False).mean().to_numpy(np.float32) for tau in TAUS]
        X = np.concatenate(feats, axis=1)
        pk = g.prog_ip_MA.cummax().to_numpy()
        ramp_ok = ~(g.prog_ip_MA.to_numpy() < 0.9 * pk)
        t = g.t.to_numpy()
        keep = on & ramp_ok
        if r.disrupted and r.t_trigger_s == r.t_trigger_s:
            keep &= t < r.t_trigger_s
        keep &= (np.arange(len(g)) % STRIDE) == 0
        out["X"].append(X[keep])
        out["risk"].append(risk[keep].astype(np.float64))
        out["lam_sim"].append(np.where(on & ramp_ok, lam_sim, 0.0)[keep])
        out["shot"].append(np.full(keep.sum(), sid))
        out["t"].append(t[keep])
        out["ramp_ok"].append(ramp_ok[keep])
    res = {k: np.concatenate(v) for k, v in out.items()}
    shots = sh.reset_index()[["shot_id", "scenario", "disrupted", "t_trigger_s", "t_tq_s", "spontaneous", "wall_contact"]]
    return res, shots


if __name__ == "__main__":
    base = Path(sys.argv[1]).expanduser()
    for name in sys.argv[2:]:
        dst = base / f"feat_{name}.npz"
        if dst.exists():
            print(name, "уже є")
            continue
        res, shots = build(base / name)
        shots.to_csv(base / f"shots_{name}.csv", index=False)
        np.savez(dst, **res)
        _, share, lam = T.decompose(res["risk"])
        dom = np.array(T.MECH)[share.argmax(1)]
        print(f"{name}: рядків {len(lam)}, λ>0,01: {(lam > 0.01).mean():.1%}, λ>0,1: {(lam > 0.1).mean():.1%}; "
              f"домінує (λ>0,01): {pd.Series(dom[lam > 0.01]).value_counts().to_dict()}", flush=True)
